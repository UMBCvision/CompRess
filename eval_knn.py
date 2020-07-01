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
from models.resnet import resnet18,resnet50
from models.alexnet import AlexNet as alexnet
from models.mobilenet import MobileNetV2 as mobilenet


parser = argparse.ArgumentParser(description='NN evaluation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', type=str, default='alexnet',
                        choices=['alexnet' , 'resnet18' , 'resnet50', 'mobilenet' ,
                                 'moco_alexnet' , 'moco_resnet18' , 'moco_resnet50', 'moco_mobilenet' ,
                                 'sup_alexnet' , 'sup_resnet18' , 'sup_resnet50', 'sup_mobilenet'])


parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='./output/cluster_alignment_1', type=str,
                    help='experiment output directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('-k', default=1, type=int, help='k in kNN')


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)


def get_model(args):

    model = None
    if args.arch == 'alexnet' :
        model = alexnet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'] , strict=False)

    elif args.arch == 'resnet18' :
        model = resnet18()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'], strict=False)

    elif args.arch == 'mobilenet' :
        model = mobilenet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'] , strict=False)

    elif args.arch == 'resnet50' :
        model = resnet50()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'], strict=False)

    elif args.arch == 'moco_alexnet' :
        model = alexnet()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q', model)]))
        model = model.cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)

    elif args.arch == 'moco_resnet18' :
        model = resnet18().cuda()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)
        model.module.encoder_q.fc = nn.Sequential()

    elif args.arch == 'moco_mobilenet' :
        model = mobilenet()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q', model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    elif args.arch == 'moco_resnet50' :
        model = resnet50().cuda()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)
        model.module.encoder_q.fc = nn.Sequential()

    elif args.arch == 'sup_alexnet' :
        model = models.alexnet(pretrained=True)
        modules = list(model.children())[:-1]
        classifier_modules = list(model.classifier.children())[:-1]
        modules.append(Flatten())
        modules.append(nn.Sequential(*classifier_modules))
        model = nn.Sequential(*modules)
        model = model.cuda()

    elif args.arch == 'sup_resnet18' :
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'sup_mobilenet' :
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'sup_resnet50' :
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    for param in model.parameters():
        param.requires_grad = False

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

    start = time.time()
    # Get train/val loader 
    # ---------------------------------------------------------------
    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers)

    # Create and load the model
    # If you want to evaluate your model, modify this part and load your model
    # ------------------------------------------------------------------------
    # MODIFY 'get_model' TO EVALUATE YOUR MODEL
    model = get_model(args)

    # ------------------------------------------------------------------------
    # Forward training samples throw the model and cache feats
    # ------------------------------------------------------------------------
    cudnn.benchmark = True

    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logger.info('load train feats from cache =>')
        train_feats, train_labels = torch.load(cached_feats)
    else:
        logger.info('get train feats =>')
        train_feats, train_labels = get_feats(train_loader, model, args.print_freq)
        torch.save((train_feats, train_labels), cached_feats)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logger.info('load val feats from cache =>')
        val_feats, val_labels = torch.load(cached_feats)
    else:
        logger.info('get train feats =>')
        val_feats, val_labels = get_feats(val_loader, model, args.print_freq)
        torch.save((val_feats, val_labels), cached_feats)

    # ------------------------------------------------------------------------
    # Calculate NN accuracy on validation set
    # ------------------------------------------------------------------------

    acc = faiss_knn(train_feats, train_labels, val_feats, val_labels, args.k)
    nn_time = time.time() - start
    logger.info('=> time : {:.2f}s'.format(nn_time))
    logger.info(' * Acc {:.2f}'.format(acc))


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

    acc = 100.0 * (pred == targets_val).mean()

    return acc


# Iterate over "loader" and forward samples through model
def get_feats(loader, model, print_freq):
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

