from collections import Counter
from random import shuffle
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import faiss

from tools import *
from models.alexnet import AlexNet
from models.mobilenet import MobileNetV2


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', default='./output/knn_1', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('-k', default=1, type=int, help='k in kNN')
parser.add_argument('--no_normalize', action='store_true', help='disable feature normalization')

best_acc1 = 0


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)


def load_weights(model, wts_path):
    if not wts_path:
        logger.info('===> no weights provided <===')
        return

    wts = torch.load(wts_path)
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    elif 'network' in wts:
        ckpt = wts['network']
    else:
        ckpt = wts

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    ckpt = {k.replace('encoder_q.', ''): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            logger.info('not copied => ' + m_key)

    model.load_state_dict(state_dict)


# 1. create a model
# 2. remove the final layer
# 3. load the weights
# 4. freeze the weights
def get_model(args):
    if args.arch == 'alexnet':
        model = AlexNet()
        model.fc = nn.Sequential()
        load_weights(model, args.weights)
        for p in model.parameters():
            p.requires_grad = False
    elif args.arch == 'conv5_alexnet':
        model = AlexNet()
        model.fc = nn.Sequential()
        load_weights(model, args.weights)
        model = nn.Sequential(
            *model.features,
            model.avgpool,
            nn.Flatten(),
        )
        print(model)
        for p in model.parameters():
            p.requires_grad = False
    elif args.arch == 'pt_alexnet':
        model = models.alexnet()
        classif = list(model.classifier.children())[:5]
        model.classifier = nn.Sequential(*classif)
        load_weights(model, args.weights)
        for p in model.parameters():
            p.requires_grad = False
    elif args.arch == 'mobilenet':
        model = MobileNetV2()
        model.fc = nn.Sequential()
        load_weights(model, args.weights)
        for p in model.parameters():
            p.requires_grad = False
    elif 'moco' in args.arch and 'alexnet' in args.arch:
        # 1. load model and weights
        key = arch_to_key[args.arch]
        model = models.__dict__[key]()
        model.fc = nn.Sequential()
        if args.weights:
            ckpt = torch.load(args.weights)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            else:
                ckpt = ckpt['state_dict']
            sd = model.state_dict()
            for k, v in ckpt.items():
                k = k.replace('module.', '')
                k = k.replace('encoder.', '')
                k = k.replace('encoder_q.', '')
                if k in sd:
                    sd[k] = v
                else:
                    print('not copied => ' + k)
            model.load_state_dict(sd)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Sequential()
    elif 'moco' in args.arch and 'resnet' in args.arch:
        # 1. load model and weights
        key = arch_to_key[args.arch]
        model = models.__dict__[key]()
        model.fc = nn.Sequential()
        if args.weights:
            ckpt = torch.load(args.weights)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            else:
                ckpt = ckpt['state_dict']
            sd = model.state_dict()
            for k, v in ckpt.items():
                k = k.replace('module.', '')
                k = k.replace('encoder.', '')
                k = k.replace('encoder_q.', '')
                if k in sd:
                    sd[k] = v
                else:
                    print('not copied => ' + k)
            model.load_state_dict(sd)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Sequential()
    elif 'resnet' in args.arch:
        key = arch_to_key[args.arch]
        model = models.__dict__[key]()
        load_weights(model, args.weights)
        for p in model.parameters():
            p.requires_grad = False

        model.fc = nn.Sequential()
    else:
        raise ValueError('arch not found: ' + args.arch)

    return model


def get_loaders(dataset_dir, bs, workers):
    # Data loading code
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # shuffle=False is very important since it is used in kmeans.py
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader


def main_worker(args):
    global best_acc1

    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers)

    model = get_model(args)
    model = nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logger.info('load train feats from cache =>')
        train_feats, train_labels = torch.load(cached_feats)
    else:
        logger.info('get train feats =>')
        train_feats, train_labels = get_feats(train_loader, model, args.print_freq, args.no_normalize)
        torch.save((train_feats, train_labels), cached_feats)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logger.info('load val feats from cache =>')
        val_feats, val_labels = torch.load(cached_feats)
    else:
        logger.info('get train feats =>')
        val_feats, val_labels = get_feats(val_loader, model, args.print_freq, args.no_normalize)
        torch.save((val_feats, val_labels), cached_feats)

    start = time.time()
    ap = faiss_knn(train_feats, train_labels, val_feats, val_labels, args.k)
    faiss_time = time.time() - start
    logger.info('=> faiss time : {:.2f}s'.format(faiss_time))
    logger.info(' * Acc {:.2f}'.format(ap))


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def faiss_knn(feats_train, targets_train, feats_val, targets_val, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    D, I = gpu_index.search(feats_val, k)

    pred = np.zeros(I.shape[0])
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i] = max(votes, key=lambda x: x[1])[0]

    ap = 100.0 * (pred == targets_val).mean()

    return ap


def get_feats(loader, model, print_freq, no_normalize):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            if no_normalize:
                cur_feats = model(images).cpu()
            else:
                cur_feats = normalize(model(images)).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == '__main__':
    main()

