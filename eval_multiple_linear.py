import re
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

from tools import *
from models.alexnet import AlexNet
from models.mobilenet import MobileNetV2


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    nargs='*', type=str,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov SGD')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output/distill_1', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', nargs='*', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--mean_paths', nargs='*', type=str, required=True,
                    help='var and mean ids of the models')
parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                    help='lr drop schedule')


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

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


def get_model(arch, wts_path):
    if arch == 'alexnet':
        model = AlexNet()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif arch == 'pt_alexnet':
        model = models.alexnet()
        classif = list(model.classifier.children())[:5]
        model.classifier = nn.Sequential(*classif)
        load_weights(model, wts_path)
    elif arch == 'mobilenet':
        model = MobileNetV2()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif 'moco' in arch and 'resnet' in arch:
        key = arch_to_key[arch]
        model = models.__dict__[key]()
        model.fc = nn.Sequential()
        if wts_path:
            ckpt = torch.load(wts_path)
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
                elif 'encoder_k.' in k:
                    continue
                else:
                    print('not copied => ' + k)
            model.load_state_dict(sd)
        model.fc = nn.Sequential()
    elif 'resnet' in arch:
        key = arch_to_key[arch]
        model = models.__dict__[key]()
        load_weights(model, wts_path)
        model.fc = nn.Sequential()
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'jig_alexnet':
        c = 256 * 6 * 6
    elif arch == 'dc_alexnet':
        c = 256 * 6 * 6
    elif arch == 'rn_alexnet':
        c = 256 * 6 * 6
    elif arch == 'resnet_moco':
        c = 2048
    elif arch == 'rotnet_r50':
        c = 1024
    elif arch == 'rotnet_r18':
        c = 256
    elif arch == 'resnet18_moco':
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    else:
        raise ValueError('arch not found: ' + arch)

    return c


class FullBatchNorm(nn.Module):
    def __init__(self, var_mean_path):
        super(FullBatchNorm, self).__init__()
        var, mean = torch.load(var_mean_path)
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


class MultipleLinearLayers(nn.Module):
    def __init__(self, arks, mean_ids, mean_paths, nc):
        super(MultipleLinearLayers, self).__init__()
        assert len(arks) == len(mean_ids)
        self.arks = arks
        channels = [get_channels(arch) for arch in arks]
        # 1. feature transform prefixes
        acc_prefixes = ['NBf']
        fmt_str = ' {:>8s}{:_>4s}'
        self.acc_prefixes = [
            fmt_str.format(mp, ap)
            for mp in mean_ids
                for ap in acc_prefixes
        ]
        # 2. actual feature transforms
        # 3. all prefix transforms are applied for each model
        self.pre_linear = nn.ModuleList([
            nn.ModuleList([
                # also change the prefixes in step 1.
                nn.Sequential(
                    Normalize(),
                    FullBatchNorm(pth),
                )
            ])
            for c, pth in zip(channels, mean_paths)
        ])
        # 4. one linear layer for each prefix transform and each model
        self.linear = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(c, nc) for _ in self.pre_linear[0]
            ])
            for c in channels
        ])

    def forward(self, inputs):
        # 1. apply each feature transform to features from each model
        pre_linear = [
            [module(inp) for module in pre_linear]
            for pre_linear, inp in zip(self.pre_linear, inputs)
        ]
        # 2. get linear layer outputs for each transformed feature
        out_linear = [
            linear_layer(inp) 
            for linear, inputs in zip(self.linear, pre_linear)
                for linear_layer, inp in zip(linear, inputs)
        ]
        return out_linear


class NoMeanEnsembleNet(nn.ModuleList):
    def forward(self, x):
        out = [m(x).detach() for m in self]
        return out


def main_worker(args):
    global best_acc1

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    arks = args.arch
    wts_paths = args.weights
    backbone = NoMeanEnsembleNet([
        get_model(arch, wts_path)
        for arch, wts_path in zip(arks, wts_paths)
    ])
    backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()

    nc = len(train_dataset.classes)
    mean_paths = args.mean_paths
    mean_ids = [re.findall(r'knn_[0-9]+', mp)[0] for mp in mean_paths]
    model = MultipleLinearLayers(arks, mean_ids, mean_paths, nc)
    acc_prefixes = model.acc_prefixes
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, backbone, model, acc_prefixes, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, model, optimizer, acc_prefixes, epoch, args)

        # evaluate on validation set
        validate(val_loader, backbone, model, acc_prefixes, args)
        # validate(val_loader, backbone, model, args)

        # modify lr
        lr_scheduler.step()
        logger.info('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))

        save_each_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, epoch, args.save)


def train(train_loader, backbone, model, optimizer, acc_prefixes, epoch, args):
    batch_time = AverageMeter('B', ':.2f')
    data_time = AverageMeter('D', ':.2f')

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            features = backbone(images)
        outputs = model(features)

        if not i:
            acc_meters = [
                NoBatchAverageMeter('', ':>11.2f')
                for i in range(len(outputs))
            ]
            progress = NoTabProgressMeter(
                len(train_loader),
                [batch_time, data_time, *acc_meters],
                prefix="Epoch: [{}]".format(epoch))

        # measure accuracy
        optimizer.zero_grad()
        for output, acc_meter in zip(outputs, acc_meters):
            loss = F.cross_entropy(output, target)
            loss.backward()
            acc1, _ = accuracy(output, target, topk=(1, 5))
            acc_meter.update(acc1[0], images.size(0))
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            line = progress.display(i)
            len_prefixes = len(acc_prefixes) * len(acc_prefixes[0])
            prefix_line = ' ' * (len(line) - len_prefixes)
            prefix_line += ''.join(acc_prefixes)
            logger.info(prefix_line)
            logger.info(line)


def validate(val_loader, backbone, model, acc_prefixes, args):
    batch_time = AverageMeter('Time', ':.3f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            features = backbone(images)
            outputs = model(features)

            if not i:
                acc_meters = [
                    NoBatchAverageMeter('', ':11.2f')
                    for i in range(len(outputs))
                ]
                progress = NoTabProgressMeter(
                    len(val_loader),
                    [batch_time, *acc_meters],
                    prefix='Test: ')

            # measure accuracy
            for output, acc_meter in zip(outputs, acc_meters):
                acc1, _ = accuracy(output, target, topk=(1, 5))
                acc_meter.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader)-1:
                line = progress.display(i)
                len_prefixes = len(acc_prefixes) * len(acc_prefixes[0])
                prefix_line = ' ' * (len(line) - len_prefixes)
                prefix_line += ''.join(acc_prefixes)
                logger.info(prefix_line)
                logger.info(line)


class NoBatchAverageMeter(object):
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
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class NoTabProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == '__main__':
    main()
