import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

from torchvision import transforms, datasets
import torch.nn as nn

from util import adjust_learning_rate, AverageMeter
from models.resnet import resnet18, resnet50
from models.alexnet import AlexNet as alexnet
from models.mobilenet import MobileNetV2 as mobilenet
from nn.compress_loss import CompReSS, Teacher

from collections import OrderedDict


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=2, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=130, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='90,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model definition
    parser.add_argument('--student_arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'])
    parser.add_argument('--teacher_arch', type=str, default='resnet50',
                        choices=['resnet50x4', 'resnet50'])
    parser.add_argument('--cache_teacher', action='store_true',
                        help='use cached teacher')

    # CompReSS loss function
    parser.add_argument('--compress_memory_size', type=int, default=128000)
    parser.add_argument('--compress_t', type=float, default=0.04)

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--teacher', type=str, help='teacher weights/feats')

    parser.add_argument('--data', type=str, help='first model')

    parser.add_argument('--checkpoint_path', default='output/', type=str,
                        help='where to save checkpoints. ')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target


# Create teacher model and load weights. For cached teacher load cahced features instead.
def get_teacher_model(opt):
    teacher = None
    if opt.cache_teacher:
        train_feats, train_labels, indices = torch.load(opt.teacher)
        teacher = Teacher(cached=True, cached_feats=train_feats)

    elif opt.teacher_arch == 'resnet50':
        model_t = resnet50()
        model_t.fc = nn.Sequential()
        model_t = nn.Sequential(OrderedDict([('encoder_q', model_t)]))
        model_t = torch.nn.DataParallel(model_t).cuda()
        checkpoint = torch.load(opt.teacher)
        model_t.load_state_dict(checkpoint['state_dict'], strict=False)
        model_t = model_t.module.cpu()

        for p in model_t.parameters():
            p.requires_grad = False
        teacher = Teacher(cached=False, model=model_t)

    return teacher


# Create student query/key model
def get_student_model(opt):
    student = None
    if opt.student_arch == 'alexnet':
        student = alexnet()
        student.fc = nn.Sequential()

    elif opt.student_arch == 'mobilenet':
        student = mobilenet()
        student.fc = nn.Sequential()

    elif opt.student_arch == 'resnet18':
        student = resnet18()
        student.fc = nn.Sequential()

    elif opt.student_arch == 'resnet50':
        student = resnet50(fc_dim=8192)

    return student


# Create train loader
def get_train_loader(opt):
    data_folder = os.path.join(opt.data, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = ImageFolderEx(
        data_folder,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader


def main():

    args = parse_option()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_loader = get_train_loader(args)

    teacher = get_teacher_model(args)
    student = get_student_model(args)

    # Calculate feature dimension of student and teacher
    teacher.eval()
    student.eval()
    tmp_input = torch.randn(2, 3, 224, 224)
    feat_t = teacher.forward(tmp_input, 0)
    feat_s = student(tmp_input)
    student_feats_dim = feat_s.shape[-1]
    teacher_feats_dim = feat_t.shape[-1]

    compress = CompReSS(teacher_feats_dim, student_feats_dim, args.compress_memory_size, args.compress_t)

    student = torch.nn.DataParallel(student).cuda()
    teacher.gpu()

    optimizer = torch.optim.SGD(student.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    args.start_epoch = 1
    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss = train_student(epoch, train_loader, teacher, student, compress, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))



        # saving the model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()



def train_student(epoch, train_loader, teacher, student, compress, optimizer, opt):
    """
    one epoch training for CompReSS
    """
    student.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, (index, inputs, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()

        # ===================forward=====================

        teacher_feats = teacher.forward(inputs, index)
        student_feats = student(inputs)

        loss = compress(teacher_feats, student_feats)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter))
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()
