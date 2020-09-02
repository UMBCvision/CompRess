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
from models.resnet_swav import resnet50w5
from eval_linear import load_weights
from eval_knn import get_feats, faiss_knn, ImageFolderEx
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
                    choices=['resnet50x4','resnet50', 'resnet50w5'])
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

    logger = get_logger(
        logpath=os.path.join(args.save, 'logs'),
        filepath=os.path.abspath(__file__)
    )
    def print_pass(*args):
        logger.info(*args)
    builtins.print = print_pass

    print(args)

    main_worker(args)


def get_model(args):
    model = None

    if args.arch == 'resnet50x4':
        model = resnet50x4()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.fc = nn.Sequential()
    elif args.arch == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Sequential()
        checkpoint = torch.load(args.weights)
        sd = checkpoint['state_dict']
        sd = {k: v for k, v in sd.items() if 'encoder_q' in k}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k.replace('module.encoder_q.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    elif args.arch == 'resnet50w5':
        model = resnet50w5()
        model.l2norm = None
        load_weights(model, args.weights)

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


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def main_worker(args):
    model = get_model(args)
    model = nn.DataParallel(model).cuda()

    train_loader, val_loader = get_data_loader(args)

    model.eval()

    cudnn.benchmark = True

    feats_file = '%s/train_feats.pth.tar' % args.save
    print('get train feats =>')
    train_feats, train_labels, train_inds = get_feats(train_loader, model, args.print_freq)
    torch.save((train_feats, train_labels, train_inds), feats_file)

    feats_file = '%s/val_feats.pth.tar' % args.save
    print('get val feats =>')
    val_feats, val_labels, val_inds = get_feats(val_loader, model, args.print_freq)
    torch.save((val_feats, val_labels, val_inds), feats_file)

    train_feats = normalize(train_feats)
    val_feats = normalize(val_feats)
    acc = faiss_knn(train_feats, train_labels, val_feats, val_labels, k=1)
    print(' * Acc {:.2f}'.format(acc))


if __name__ == '__main__':
    main()
