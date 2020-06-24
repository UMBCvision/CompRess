import argparse
import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import faiss

from tools import *
from eval_knn import get_model, get_loaders, normalize


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='./output/kmeans_1', type=str,
                    help='experiment output directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--clusters', default=2000, type=int, help='numbe of clusters')

best_acc1 = 0


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)


def main_worker(args):
    global best_acc1

    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers)

    model = get_model(args)
    model = model.cuda()

    cudnn.benchmark = True

    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logger.info('load train feats from cache =>')
        train_feats, _ = torch.load(cached_feats)
    else:
        logger.info('get train feats =>')
        train_feats, _ = get_feats(train_loader, model, args.print_freq)
        torch.save((train_feats, _), cached_feats)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logger.info('load val feats from cache =>')
        val_feats, _ = torch.load(cached_feats)
    else:
        logger.info('get val feats =>')
        val_feats, _ = get_feats(val_loader, model, args.print_freq)
        torch.save((val_feats, _), cached_feats)

    start = time.time()
    train_a, val_a = faiss_kmeans(train_feats, val_feats, args.clusters)

    samples = list(s.replace(args.data + '/train/', '') for s, _ in train_loader.dataset.samples)
    train_s = list((s, a) for s, a in zip(samples, train_a))
    train_d_path = os.path.join(args.save, 'train_clusters.txt')
    with open(train_d_path, 'w') as f:
        for pth, cls in train_s:
            f.write('{} {}\n'.format(pth, cls))

    samples = list(s.replace(args.data + '/val/', '') for s, _ in val_loader.dataset.samples)
    val_s = list((s, a) for s, a in zip(samples, val_a))
    val_d_path = os.path.join(args.save, 'val_clusters.txt')
    with open(val_d_path, 'w') as f:
        for pth, cls in val_s:
            f.write('{} {}\n'.format(pth, cls))

    faiss_time = time.time() - start
    logger.info('=> faiss time : {:.2f}s'.format(faiss_time))


def faiss_kmeans(train_feats, val_feats, nmb_clusters):
    train_feats = train_feats.numpy()
    val_feats = val_feats.numpy()

    d = train_feats.shape[-1]

    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    index = faiss.IndexFlatL2(d)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    index = faiss.index_cpu_to_all_gpus(index, co)

    # perform the training
    clus.train(train_feats, index)
    _, train_a = index.search(train_feats, 1)
    _, val_a = index.search(val_feats, 1)

    return list(train_a[:, 0]), list(val_a[:, 0])


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
