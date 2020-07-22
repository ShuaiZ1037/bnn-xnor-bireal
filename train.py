#-*-coding:utf-8-*-
import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

#sys.path.append("../")
from utils import *
import torchvision
from torchvision import datasets, transforms
from binary_model import resnet_bireal, resnet_bnn, resnet_xnor

torchvision.set_image_backend('accimage')


parser = argparse.ArgumentParser("train")
parser.add_argument('--binary_method', type=str, default='bireal', help='the binary method: bnn, xnor or bireal')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--print-interval', type=int, default=20, help='how many batches to wait before print training status')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--save', type=str, default='./cifar10', help='path for saving trained models')
parser.add_argument('--resume', default=False, help='resume from the checkpoint')
parser.add_argument('--data', default='/media/zxc/FILE/ZS/data',metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.01, help='label smoothing')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

CLASSES = 10
save_root = './result/' + args.binary_method
if not os.path.exists(save_root):
    os.makedirs(save_root)

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True
    print("args = %s", args)

    # load model
    if args.binary_method == 'bireal':
        model = resnet_bireal.birealnet20()
    elif args.binary_method == 'xnor':
        model = resnet_xnor.xnornet20()   # xnor
    elif args.binary_method == 'bnn':
        model = resnet_bnn.bnn20()    # bnn
    else:
        print('Error binary method!')
    model = model.cuda()
    # model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    # criterion
    criterion_smooth = criterion_smooth.cuda()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
            weight_parameters.append(p)
            print(pname)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    optimizer = torch.optim.Adam(
            [{'params': other_parameters, 'weight_decay': args.weight_decay},
            {'params': weight_parameters, 'weight_decay': args.weight_decay}],
            lr=args.learning_rate,)
    # optimizer = torch.optim.SGD(
    #     [{'params': other_parameters, 'weight_decay': args.weight_decay},
    #      {'params': weight_parameters, 'weight_decay': args.weight_decay}],
    #     lr=args.learning_rate, momentum=args.momentum)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 140, 180, 220], gamma=0.1)
    start_epoch = 0
    best_top1_acc = 0

    # resume
    if args.resume:
        checkpoint_tar = os.path.join(save_root, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_tar):
            print('loading checkpoint {} ..........'.format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            start_epoch = checkpoint['epoch']
            best_top1_acc = checkpoint['best_top1_acc']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
        else:
            raise ValueError('There is no checkpoint.')

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transforms_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train(epoch, train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, model, criterion, args)


        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, save_root)

        epoch += 1

    training_time = (time.time() - start_t) / 3600
    print('total training time = {} hours. best acc: {}'.format(training_time, best_top1_acc))


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))


    model.train()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('epoch: %d base learning_rate: %e' % (epoch, cur_lr))

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, = accuracy(logits, target, topk=(1,))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0:
            progress.display(i)

    # return losses.avg, top1.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, = accuracy(logits, target, topk=(1,))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_interval == 0:
                progress.display(i)

        print(' * acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return losses.avg, top1.avg


if __name__ == '__main__':
    for i in range(4):
        main()
