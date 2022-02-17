import os
import argparse
import pathlib
import random
import shutil
import sys
import time
import warnings

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
#tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        #tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
        
import wandb

from data_wrapper import KWSDataWrapper
import get_dataset as kws_data
import kws_util
import models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Google-Speech-Commands Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ds_cnn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: ds_cnn)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# MR
parser.add_argument('-d', '--dataset', default='GoogleSpeechCommands', type=str,
                    help='GoogleSpeechCommands')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step-epoch', default=30, type=int, metavar='N',
                    help='number of epochs to decay learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch-cfg', '--ac', default='', type=str, metavar='PATH',
                    help='path to architecture configuration')
# MR
parser.add_argument('-ft', '--fine-tune', dest='fine_tune', action='store_true',
                    help='use pre-trained weights from search phase')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--visualization', dest='visualization', action='store_true',
                    help='visualize training logs using wandb')
parser.add_argument('-pr', '--project', default='misc', type=str,
                    help='wandb project name')
parser.add_argument('--tags', nargs='+', default=None,
                    help='wandb tags')


best_acc1 = 0
best_acc1_test = 0

def main():
    args = parser.parse_args()
    print(args)

    complexity_decay = args.data.split('_')[-1]

    if args.visualization:
        wandb.init(
            project = args.project,
            entity = 'matteorisso',
            name = f'Fine-Tune: {complexity_decay}',
            notes = f'Fine-Tune arch found with {complexity_decay} strength',
            tags = ['Fine-Tune', args.arch] + args.tags,
            dir = args.data
        )
        wandb.config.update(args)
        wandb.define_metric('Train/Loss', summary='min')
        wandb.define_metric('Train/Acc', summary='max')
        wandb.define_metric('Test/Loss', summary='min')
        wandb.define_metric('Test/Acc', summary='max')


    args.data = pathlib.Path(args.data)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc1_test
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    if args.dataset == 'GoogleSpeechCommands':
        num_classes = 12

        data_dir = args.data.parent.parent.parent / 'GoogleSpeechCommands'

        Flags, unparsed = kws_util.parse_command()
        Flags.data_dir = str(data_dir)
        Flags.bg_path = str(data_dir)
        #Flags.batch_size = 1
        print(f'We will download data to {Flags.data_dir}')
        ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
        print("Done getting data")

        train_shuffle_buffer_size = 85511
        val_shuffle_buffer_size = 10102
        test_shuffle_buffer_size = 4890

        ds_train = list(ds_train.shuffle(train_shuffle_buffer_size).as_numpy_iterator())
        x_train, y_train = [], []
        for x, y in ds_train:
            x_train.append(x)
            y_train.append(np.expand_dims(y, axis=1))
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train).squeeze(-1)
        inp_shape = x_train.shape
        x_train = QuantileTransformer().fit_transform(
            np.expand_dims(x_train.ravel(), -1)
            ).reshape(inp_shape)
        #x_train = RobustScaler(quantile_range=(1,99)).fit_transform(
        #    np.expand_dims(x_train.ravel(), -1)
        #    ).reshape(inp_shape)
        #x_train = MinMaxScaler((0,6)).fit_transform(
        #    np.expand_dims(x_train.ravel(), -1)
        #    ).reshape(inp_shape)

        ds_val = list(ds_val.shuffle(val_shuffle_buffer_size).as_numpy_iterator())
        x_val, y_val = [], []
        for x, y in ds_val:
            x_val.append(x)
            y_val.append(np.expand_dims(y, axis=1))
        x_val = np.vstack(x_val)
        y_val = np.vstack(y_val).squeeze(-1)
        inp_shape = x_val.shape
        x_val = QuantileTransformer().fit_transform(
            np.expand_dims(x_val.ravel(), -1)
            ).reshape(inp_shape)
        #x_val = RobustScaler(quantile_range=(1,99)).fit_transform(
        #    np.expand_dims(x_val.ravel(), -1)
        #    ).reshape(*inp_shape)
        #x_val = MinMaxScaler((0,6)).fit_transform(
        #    np.expand_dims(x_val.ravel(), -1)
        #    ).reshape(*inp_shape)

        ds_test = list(ds_test.shuffle(test_shuffle_buffer_size).as_numpy_iterator())
        x_test, y_test = [], []
        for x, y in ds_test:
            x_test.append(x)
            y_test.append(np.expand_dims(y, axis=1))
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test).squeeze(-1)
        inp_shape = x_test.shape
        x_test = QuantileTransformer().fit_transform(
            np.expand_dims(x_test.ravel(), -1)
            ).reshape(inp_shape)
        #x_test = RobustScaler(quantile_range=(1,99)).fit_transform(
        #    np.expand_dims(x_test.ravel(), -1)
        #    ).reshape(*inp_shape)
        #x_test = MinMaxScaler((0,6)).fit_transform(
        #    np.expand_dims(x_test.ravel(), -1)
        #    ).reshape(*inp_shape)
        
        #train_set = KWSDataWrapper(data_generator=ds_train)
        #val_set = KWSDataWrapper(data_generator=ds_val)
        #test_set = KWSDataWrapper(data_generator=ds_test)
        train_set = KWSDataWrapper(x_train, y_train)
        val_set = KWSDataWrapper(x_val, y_val)
        test_set = KWSDataWrapper(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=100, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=100, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        #x_train = np.stack([_[0].squeeze(0) 
        #    for _ in ds_train.shuffle(train_shuffle_buffer_size).as_numpy_iterator()
        #    ], axis=0)
        #inp_shape = x_train.shape[1:]
        #x_train = RobustScaler(quantile_range=(1,99)).fit_transform(
        #    np.expand_dims(x_train.ravel(), -1)
        #    ).reshape(-1, *inp_shape)
        #x_train = MinMaxScaler((0,6)).fit_transform(
        #    np.expand_dims(x_train.ravel(), -1)
        #    ).reshape(-1, *inp_shape)
        #y_train = np.stack([_[1]
        #    for _ in ds_train.shuffle(train_shuffle_buffer_size).as_numpy_iterator()
        #    ], axis=0)
        #train_set = KWSDataWrapper(x_train, y_train)
        #if args.distributed:
        #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #else:
        #    train_sampler = None
        #train_loader = torch.utils.data.DataLoader(
        #    train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        #    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        #x_val = np.stack([_[0].squeeze(0) 
        #    for _ in ds_val.shuffle(val_shuffle_buffer_size).as_numpy_iterator()
        #    ], axis=0)
        #inp_shape = x_val.shape[1:]
        #x_val = RobustScaler(quantile_range=(1,99)).fit_transform(
        #    np.expand_dims(x_val.ravel(), -1)
        #    ).reshape(-1, *inp_shape)
        #x_val = MinMaxScaler((0,6)).fit_transform(
        #    np.expand_dims(x_val.ravel(), -1)
        #    ).reshape(-1, *inp_shape)
        #y_val = np.stack([_[1]
        #    for _ in ds_val.shuffle(val_shuffle_buffer_size).as_numpy_iterator()
        #    ], axis=0) 
        #val_set = KWSDataWrapper(x_val, y_val)
        #val_loader = torch.utils.data.DataLoader(
        #    val_set, batch_size=args.batch_size, shuffle=False,
        #    num_workers=args.workers, pin_memory=True)
        
        #x_test = np.stack([_[0].squeeze(0) 
        #    for _ in ds_test.shuffle(test_shuffle_buffer_size).as_numpy_iterator()
        #    ], axis=0)
        #inp_shape = x_test.shape[1:]
        #x_test = RobustScaler(quantile_range=(1,99)).fit_transform(
        #    np.expand_dims(x_test.ravel(), -1)
        #    ).reshape(-1, *inp_shape)
        #x_test = MinMaxScaler((0,6)).fit_transform(
        #    np.expand_dims(x_test.ravel(), -1)
        #    ).reshape(-1, *inp_shape)
        #y_test = np.stack([_[1]
        #    for _ in ds_test.shuffle(test_shuffle_buffer_size).as_numpy_iterator()
        #    ], axis=0) 
        #test_set = KWSDataWrapper(x_test, y_test)
        #test_loader = torch.utils.data.DataLoader(
        #    test_set, batch_size=args.batch_size, shuffle=False,
        #    num_workers=args.workers, pin_memory=True)
    else:
        raise ValueError('Unknown dataset: {}. Please use "GoogleSpeechCommands" instead"'.format(args.dataset)) 
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    if len(args.arch_cfg) > 0:
        if os.path.isfile(args.arch_cfg):
            print("=> loading architecture config from '{}'".format(args.arch_cfg))
        else:
            print("=> no architecture found at '{}'".format(args.arch_cfg))
    model = models.__dict__[args.arch](args.arch_cfg, num_classes=num_classes, fine_tune=args.fine_tune)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if 'alex' in args.arch or 'vgg' in args.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    #criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, 
                                    weight_decay=args.weight_decay)
    
    # Alternative scheduling from https://github.com/kuangliu/pytorch-cifar
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    
    # Data loading code
    #quantres18_w248a248_multiprectraindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    #if 'inception' in args.arch:
    #    crop_size, short_size = 299, 342
    #else:
    #    crop_size, short_size = 224, 256
    #train_dataset = datasets.ImageFolder(
    #    traindir,
    #    transforms.Compose([
    #        transforms.RandomResizedCrop(crop_size),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))

    #if args.distributed:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #else:
    #    train_sampler = None

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    #val_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        transforms.Resize(short_size),
    #        transforms.CenterCrop(crop_size),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = args.start_epoch
    best_epoch_test = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Original lr scheduling
        adjust_learning_rate(optimizer, epoch, args)

        # Alternative scheduling from https://github.com/kuangliu/pytorch-cifar
        #scheduler.step()

        # train for one epoch
        #import pdb; pdb.set_trace()
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)
        acc1_test = validate(test_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        is_best_test = acc1_test > best_acc1_test
        best_acc1 = max(acc1, best_acc1)
        best_acc1_test = max(acc1_test, best_acc1_test)
        if is_best:
            best_epoch = epoch
        if is_best_test:
            best_epoch_test = epoch

        #print('========= architecture info =========')
        #if hasattr(model, 'module'):
        #    bitops, bita, bitw = model.module.fetch_arch_info()
        #else:
        #    bitops, bita, bitw = model.fetch_arch_info()
        #print('model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(bitops, bita, bitw))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(args.data, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch, args.step_epoch)

    best_acc1_val = best_acc1 
    print('Best Acc_val@1 {0} @ epoch {1}'.format(best_acc1_val, best_epoch))

    test_acc1 = best_acc1_test
    print('Test Acc_val@1 {0} @ epoch {1}'.format(test_acc1, best_epoch_test))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t".format(epoch, args.epochs, curr_lr))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        images = images.squeeze(0)
        target = target.squeeze(0)

        # compute output
        output = model(images.transpose(1,3).transpose(2,3))
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    # Visualization
    if args.visualization:
        wandb.log({
                "Epoch": epoch,
                "Train/Loss": losses.avg, 
                "Train/Acc": top1.avg,
                "Train/lr": curr_lr
            })

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            images = images.squeeze(0)
            target = target.squeeze(0)

            # compute output
            output = model(images.transpose(1,3).transpose(2,3))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {:.3f}'
              .format(top1.avg))

    # Visualization
    if args.visualization:
        wandb.log({
                "Epoch": epoch,
                "Test/Loss": losses.avg, 
                "Test/Acc": top1.avg
            })

    return top1.avg


def save_checkpoint(root, state, is_best, epoch, step_epoch, filename='checkpoint.pth.tar'):
    torch.save(state, root / filename)
    if is_best:
        shutil.copyfile(root / filename, root / 'model_best.pth.tar')
    if (epoch + 1) % step_epoch == 0:
        shutil.copyfile(root / filename, root / 'checkpoint_ep{}.pth.tar'.format(epoch + 1))


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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
    #lr = args.lr * (0.1 ** (epoch // args.step_epoch))
    if epoch < 50:
        args.lr = 1e-2
    elif epoch < 100:
        args.lr = 5e-3
    elif epoch < 150:
        args.lr = 2.5e-3
    else:
        args.lr = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()