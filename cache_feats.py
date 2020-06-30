import argparse
import builtins
import os
import random
import shutil
import time
import warnings
from collections import Counter
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tools import *
from models.resnet50x4 import Resnet50_X4 as resnet50x4
import numpy as np
import time
import pickle


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--arch', type=str, default='resnet50x4',
                    choices=['resnet50x4','resnet50'])
parser.add_argument('--data_pre_processing', type=str, default='SimCLR',
                    choices=['SimCLR','MoCo'])
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', type=str, default='output/cached_feats',
                    help='directory to store cached features')
parser.add_argument('--weights', default='', type=str,
                    help='path to pretrained model checkpoint')


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)


class ImageFolderEx(datasets.ImageFolder) :

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index , sample, target


def get_model(args):
    model = None

    if args.arch == 'resnet50x4':
        model = resnet50x4().cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    elif args.arch == 'resnet50':
        model = models.resnet50().cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    for p in model.parameters():
        p.requires_grad = False

    return model


def get_data_loader(args):
    if args.data_pre_processing == 'SimCLR':
        # Data loaders
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_loader = torch.utils.data.DataLoader(
            ImageFolderEx(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFolderEx(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.data_pre_processing == "MoCo":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        # Data loaders
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_loader = torch.utils.data.DataLoader(
            ImageFolderEx(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFolderEx(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def main_worker(args):

    model = get_model(args)

    train_loader, val_loader = get_data_loader(args)

    model.eval()

    cudnn.benchmark = True

    print('Test Model =>')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (_, images, targets) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % 20 == 19:
                print(' [{0}/{1}]\t'  
                      'Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(i+1 , len(val_loader), top1=top1, top5=top5))

        print("Accuracy on Validation set => ")
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    model.fc = nn.Sequential()

    cached_feats = '%s/train_feats.pth.tar' % args.save
    print('Get Features =>')
    train_feats, train_labels , indices = get_feats(train_loader, model, args.print_freq)
    torch.save((train_feats, train_labels , indices), cached_feats)


def get_feats(loader, model, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels,indices, ptr = None, None , None , 0

    with torch.no_grad():
        end = time.time()
        for i, (index, images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = model(images).cpu()
            cur_indices = index.cpu()

            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()
                indices = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            indices.index_copy_(0 , inds , cur_indices)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels , indices


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
