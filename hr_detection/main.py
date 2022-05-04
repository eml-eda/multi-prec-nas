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

from dataset import split_train_test
import models as models
from loss import LogCoshLoss

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch HR Detection Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='temponet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: temponet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# MR
parser.add_argument('-d', '--dataset', default='PPG_Dalia', type=str,
                    help='PPG_Dalia')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--patience', default=20, type=int, metavar='N',
                    help='number of epochs wout improvements to wait before early stopping')
parser.add_argument('--step-epoch', default=25, type=int, metavar='N',
                    help='number of epochs to decay learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
        rng = np.random.default_rng(seed=42)
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
        main_worker(args.gpu, ngpus_per_node, rng, args)


def main_worker(gpu, ngpus_per_node, rng, args):
    global best_mae
    best_mae = np.inf
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
    if args.dataset == 'PPG_Dalia':
        data_dir = args.data.parent.parent / 'data'

        train_set, test_set = split_train_test(data_dir, rng)

        # Split dataset into train and validation
        train_len = int(len(train_set) * 0.8)
        val_len = len(train_set) - train_len
        # Fix generator seed for reproducibility
        data_gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_len, val_len], generator=data_gen)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        raise ValueError('Unknown dataset: {}. Please use "PPG_Dalia" instead"'.format(args.dataset)) 
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    if len(args.arch_cfg) > 0:
        if os.path.isfile(args.arch_cfg):
            print("=> loading architecture config from '{}'".format(args.arch_cfg))
        else:
            print("=> no architecture found at '{}'".format(args.arch_cfg))
    model = models.__dict__[args.arch](args.arch_cfg, fine_tune=args.fine_tune)

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
    criterion = LogCoshLoss().cuda(args.gpu)

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
    
    if args.evaluate:
        acc1 = validate(val_loader, model, criterion, 0, args)
        acc1_test = validate(test_loader, model, criterion, 0, args)
        print(f'Acc_val: {acc1} Acc_test: {acc1_test}')
        return

    best_epoch = args.start_epoch
    best_epoch_test = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on val and test sets
        mae_val = validate(val_loader, model, criterion, epoch, args, scope='Val')
        mae_test = validate(test_loader, model, criterion, epoch, args, scope='Test')

        # remember best acc@1 and save checkpoint
        is_best = mae_val < best_mae
        if is_best:
            best_epoch = epoch
            best_mae = min(mae_val, best_mae)
            best_mae_test = mae_test
            epoch_wout_improve = 0
            print(f'New best MAE_val: {best_mae}')
        else:
            epoch_wout_improve += 1
            print(f'No improvement in {epoch_wout_improve} epochs.')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(args.data, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_mae': best_mae_test,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch, args.step_epoch)

        # Early-Stop
        if epoch_wout_improve >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print('Val MAE: {0} @ epoch {1}'.format(best_mae, best_epoch))
    print('Test MAE: {0} @ epoch {1}'.format(best_mae_test, best_epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mae = AverageMeter('MAE', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mae],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t".format(epoch, args.epochs, curr_lr))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        metr = nn.L1Loss()(output, target)
        losses.update(loss.item(), data.size(0))
        mae.update(metr.item(), data.size(0))

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
                "Train/Acc": mae.avg,
                "Train/lr": curr_lr
            })

def validate(val_loader, model, criterion, epoch, args, scope='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mae = AverageMeter('MAE', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mae],
        prefix=f'{scope}: ')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            metr = nn.L1Loss()(output, target)
            losses.update(loss.item(), data.size(0))
            mae.update(metr.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * MAE {:.3f}'
              .format(mae.avg))

    # Visualization
    if args.visualization:
        wandb.log({
                "Epoch": epoch,
                "Test/Loss": losses.avg, 
                "Test/Acc": mae.avg
            })

    return mae.avg


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

if __name__ == '__main__':
    main()