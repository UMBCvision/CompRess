import shutil

import logging
import os

import torch
from torch import nn
from torchvision import models


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_each_checkpoint(state, epoch, save_dir):
    ckpt_path = os.path.join(save_dir, 'ckpt_%d.pth.tar' % epoch)
    torch.save(state, ckpt_path)


def save_checkpoint(state, is_best, save_dir):
    ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    if is_best:
        best_ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(ckpt_path, best_ckpt_path)


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


arch_to_key = {
    'alexnet': 'alexnet',
    'alexnet_moco': 'alexnet',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50',
    'rotnet_r50': 'resnet50',
    'rotnet_r18': 'resnet18',
    'resnet18_moco': 'resnet18',
    'resnet_moco': 'resnet50',
}

model_names = list(arch_to_key.keys())


def remove_dropout(model):
    classif = model.classifier.children()
    classif = [nn.Sequential() if isinstance(m, nn.Dropout) else m for m in classif]
    model.classifier = nn.Sequential(*classif)


# 1. stores a list of models to ensemble
# 2. forward through each model and save the output
# 3. return mean of the outputs along the class dimension
class EnsembleNet(nn.ModuleList):
    def forward(self, x):
        out = [m(x) for m in self]
        out = torch.stack(out, dim=-1)
        out = out.mean(dim=-1)
        return out
