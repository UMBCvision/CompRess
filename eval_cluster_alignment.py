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
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import pickle
from models.resnet import resnet18,resnet50
from models.alexnet import AlexNet as alexnet
from models.mobilenet import MobileNetV2 as mobilenet
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
from PIL import Image
from tools import *
import random
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Cluster Alignment evaluation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--model', type=str, default='alexnet',
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
parser.add_argument('--clusters', default=1000, type=int, help='numbe of clusters')

parser.add_argument('--visualization', action='store_true',
                    help='save visualization')

parser.add_argument('--confusion_matrix', action='store_true',
                    help='save confusion matrix')

parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')

class ImageFolderEx(datasets.ImageFolder) :

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return sample, target , index



def get_loaders(dataset_dir, bs, workers):

    # Data loading code
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolderEx(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    val_loader = DataLoader(
        ImageFolderEx(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader





def get_featureEextractor(model):
    modules = list(model.children())[:-2]
    modules.append(Flatten())

    fexModel = nn.Sequential(*modules)
    return fexModel


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)


class ImageFileDataset(datasets.VisionDataset):
    def __init__(self, root, f_path, transform=None):
        super(ImageFileDataset, self).__init__(root, transform=transform)
        with open(f_path, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            lines = [(os.path.join(root, pth), int(cid)) for pth, cid in lines]
        self.samples = lines
        self.classes = sorted(set(s[1] for s in self.samples))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path)
        sample = self.transform(sample)

        return sample, target , index

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def calculate_alligment_score(inputSet , targetSet) :
    return len(Intersection(inputSet, targetSet)) / len(inputSet)

def calculate_precision(inputSet , targetSet) :
    return len(Intersection(inputSet, targetSet)) / len(inputSet)

def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

def Union(lst1, lst2):
    return set(lst1).union(lst2)

def get_model(args):

    if args.model == 'alexnet' :
        model = alexnet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'] , strict=False)

    elif args.model == 'resnet18' :
        model = resnet18()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'], strict=False)

    elif args.model == 'mobilenet' :
        model = mobilenet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'] , strict=True)

    elif args.model == 'resnet50' :
        model = resnet50()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'], strict=False)

    elif args.model == 'moco_alexnet' :
        model = alexnet()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q', model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)

    elif args.model == 'moco_resnet18' :
        model = resnet18().cuda()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)
        model.module.encoder_q.fc = nn.Sequential()

    elif args.model == 'moco_mobilenet' :
        model = mobilenet()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q', model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    elif args.model == 'moco_resnet50' :
        model = resnet50().cuda()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)
        model.module.encoder_q.fc = nn.Sequential()

    elif args.model == 'sup_alexnet' :
        model = models.alexnet(pretrained=True)
        modules = list(model.children())[:-1]
        classifier_modules = list(model.classifier.children())[:-1]
        modules.append(Flatten())
        modules.append(nn.Sequential(*classifier_modules))
        model = nn.Sequential(*modules)
        model = torch.nn.DataParallel(model).cuda()

    elif args.model == 'sup_resnet18' :
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    elif args.model == 'sup_mobilenet' :
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    elif args.model == 'sup_resnet50' :
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()




    for param in model.parameters():
        param.requires_grad = False

    return model


def train_kmeans(model , train_loader , val_loader , args):
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


def get_cluster_samples_train(train_clusters, args):
    traindir = os.path.join(args.data, 'train')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset_input = ImageFileDataset(
        traindir,
        train_clusters,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader_input = DataLoader(
        train_dataset_input, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    train_loader_target, _ = get_loaders(args.data, args.batch_size, args.workers)

    input_cluster_samples_train = []
    target_cluster_samples_train = []

    print("Creating Clusters Index Set - Training data (Target) ...")
    for i in range(1000):
        input_cluster_samples_train.append([])
        target_cluster_samples_train.append([])

    for i, (_, target, index) in enumerate(train_loader_target):
        index = index.cpu().numpy()
        for j, t in enumerate(target):
            target_cluster_samples_train[t].append(index[j])

        if i % 100 == 0:
            print("[%d/%d]" % (i, len(train_loader_target)))

    print("Creating Clusters Index Set - Training data (Input) ...")
    for i, (_, target, index) in enumerate(train_loader_input):
        index = index.cpu().numpy()
        for j, t in enumerate(target):
            input_cluster_samples_train[t].append(index[j])

        if i % 100 == 0:
            print("[%d/%d]" % (i, len(train_loader_input)))


    return input_cluster_samples_train , target_cluster_samples_train



def get_cluster_samples_val(val_clusters, args):
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader_input = torch.utils.data.DataLoader(
        ImageFileDataset(
            valdir,
            val_clusters,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    _, val_loader_target = get_loaders(args.data, args.batch_size, args.workers)
    input_cluster_samples_val = []
    target_cluster_samples_val = []
    for i in range(1000):
        input_cluster_samples_val.append([])
        target_cluster_samples_val.append([])

    print("Creating Clusters Index Set - Val data (Target) ...")
    for i, (_, target, index) in enumerate(val_loader_target):
        index = index.cpu().numpy()
        for j, t in enumerate(target):
            target_cluster_samples_val[t].append(index[j])

        if i % 100 == 0:
            print("[%d/%d]" % (i, len(val_loader_target)))

    print("Creating Clusters Index Set - Val data (Input) ...")
    for i, (_, target, index) in enumerate(val_loader_input):
        index = index.cpu().numpy()
        for j, t in enumerate(target):
            input_cluster_samples_val[t].append(index[j])

        if i % 100 == 0:
            print("[%d/%d]" % (i, len(val_loader_input)))

    return input_cluster_samples_val, target_cluster_samples_val

def dump_random_cluster(cluster_set , dataset, args) :
    cluster_index = random.randint(0,1000)
    cluster = cluster_set[cluster_index]

    selected_sample = random.sample(cluster , 20)
    imgs_list = [dataset[i] for i in selected_sample]

    result_img_list = [imgs_list[i][0] for i in range(20)]


    result_img = torch.cat(result_img_list , dim=2)

    return result_img

def dump_visualization(input_cluster_samples_train , args):
    traindir = os.path.join(args.data, 'train')
    train_dataset = ImageFolderEx(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    img_list_per_cluster = [
        dump_random_cluster(input_cluster_samples_train, train_dataset, args) for i in range(35)]

    query_img = torch.cat(img_list_per_cluster, dim=1)
    save_dir = '%s/query_img.jpg' % (args.save)
    save_image(query_img, save_dir)

def main_worker(args):

    # Get train/val loader
    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers)

    # Create and load the model
    # If you want to evaluate your model, modify this part and load your model
    # ------------------------------------------------------------------------
    # MODIFY 'get_model' TO EVALUATE YOUR MODEL
    model = get_model(args)

    # ------------------------------------------------------------------------




    # Train K-means
    # ---------------------------------------------------------------
    train_clusters = os.path.join(args.save, 'train_clusters.txt')
    val_clusters = os.path.join(args.save, 'val_clusters.txt')

    if (not os.path.exists(train_clusters)) or (not os.path.exists(val_clusters)):
        train_kmeans(model, train_loader, val_loader, args)



    cached_cluster_samples = '%s/cluster_list_cached.p' % args.save
    if os.path.exists(cached_cluster_samples):
        input_cluster_samples_train, target_cluster_samples_train = pickle.load( open(cached_cluster_samples, "rb" ) )
    else :
        input_cluster_samples_train , target_cluster_samples_train = get_cluster_samples_train(train_clusters, args)
        pickle.dump((input_cluster_samples_train, target_cluster_samples_train), open(cached_cluster_samples, "wb"))


    # ---------------------------------------------------------------







    # Map clusters to categories
    # ---------------------------------------------------------------
    cluster_allignment_cost = np.zeros((1000, 1000))
    confusion_matrix = np.zeros((1000, 1000))

    print("Calculating (input , target) Cluster pairs scores ... ")
    for i, inputSet in enumerate(input_cluster_samples_train):
        # start = time.time()
        for j, targetSet in enumerate(target_cluster_samples_train):
            confusion_matrix[i][j] = len(Intersection(inputSet, targetSet))
            cluster_allignment_cost[i][j] = -calculate_alligment_score(inputSet, targetSet)
        if i % 100 == 0:
            print("[%d/1000]" % i)

    row_ind, col_ind = linear_sum_assignment(cluster_allignment_cost)
    input_to_target_map = col_ind
    target_to_input_map = np.zeros(1000)
    for k in range(1000):
        target_to_input_map[input_to_target_map[k]] = k


    confusion_matrix = confusion_matrix[target_to_input_map.astype(int)]

    if args.confusion_matrix :
        cached_data_confusion_matrix = '%s/confusion_matrix.p' % args.save
        pickle.dump(confusion_matrix, open(cached_data_confusion_matrix, "wb"))


    if args.visualization :
        dump_visualization(input_cluster_samples_train , args)

    # ---------------------------------------------------------------



    # Calculate accuracy on train/val set
    # ---------------------------------------------------------------
    correct = 0
    all = 0
    for c in range(1000):
        input_cluster = input_cluster_samples_train[c]
        target_cluster = target_cluster_samples_train[input_to_target_map[c]]
        correct += len(Intersection(input_cluster, target_cluster))
        all += len(target_cluster)


    print("acc train : %f" % (correct / all))


    input_cluster_samples_val, target_cluster_samples_val = get_cluster_samples_val(val_clusters, args)


    correct = 0
    all = 0
    for c in range(1000):
        input_cluster = input_cluster_samples_val[c]
        target_cluster = target_cluster_samples_val[input_to_target_map[c]]
        correct += len(Intersection(input_cluster, target_cluster))
        all += len(target_cluster)

    print("acc val : %f" % (correct / all))

    # ---------------------------------------------------------------











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


    clus.train(train_feats, index)
    _, train_a = index.search(train_feats, 1)
    _, val_a = index.search(val_feats, 1)

    return list(train_a[:, 0]), list(val_a[:, 0])

def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


import pdb

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
        for i, (images, target, index) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            # pdb.set_trace()
            cur_feats = normalize(model(images)).cpu()
            # cur_feats = model(images).cpu()
            # pdb.set_trace()
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
