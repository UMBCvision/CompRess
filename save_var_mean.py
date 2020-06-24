import argparse

import torch
import numpy as np

from tools import *


parser = argparse.ArgumentParser(description='Save variance and mean of features')
parser.add_argument('--x_root', default='output/knn_22_dc_alexnet', type=str,
        help='directory containing x features')


def main():
    args = parser.parse_args()
    x = {
        'train': list(torch.load(args.x_root + '/train_feats.pth.tar')),
        'valid': list(torch.load(args.x_root + '/val_feats.pth.tar')),
    }

    x_var, x_mean = torch.var_mean(x['train'][0], dim=0)
    torch.save((x_var, x_mean), '%s/var_mean.pth.tar' % args.x_root)


if __name__ == '__main__':
    main()

