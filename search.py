import argparse
import os
import pathlib
import random
import shutil
import time
import warnings

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

import pdb

import wandb

import models as models

# Simply parse all models' names contained in models directory
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-d', '--dataset', default='None', type=str,
                    help='cifar or imagenet')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step-epoch', default=30, type=int, metavar='N',
                    help='number of epochs to decay learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup', default=0, type=int,
                    help='number of warmup epochs'
                        '(default: 0 -> no warmup)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lra', '--learning-rate-alpha', default=0.01, type=float,
                    metavar='LR', help='initial alpha learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--ad', '--alpha-decay', default=1e-4, type=float,
                    metavar='A', help='alpha decay (default: 1e-4)',
                    dest='alpha_decay')
alpha_initializations = ['same', 'scaled']
parser.add_argument('--alpha-init', '--ai', type=str,
                    choices=alpha_initializations,
                    help=f'alpha initialization method: {*alpha_initializations,}')
parser.add_argument('--complexity-decay', '--cd', default=0, type=float,
                    metavar='W', help='complexity decay (default: 0)')
regularizer_targets = ['ops', 'weights']
parser.add_argument('--regularization-target', '--rt', type=str,
                    choices=regularizer_targets,
                    help=f'regularization target: {*regularizer_targets,}')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
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
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='enable additional visualizations useful for debugging')

best_acc1 = 0

def main():
    args = parser.parse_args()
    print(args)

    if args.visualization:
        wandb.init(
            project = args.project,
            entity = 'matteorisso',
            name = f'Search: {args.complexity_decay}',
            notes = f'Multi-Precision Search with {args.complexity_decay} strength',
            tags = ['Search', args.arch] + args.tags,
            dir = args.data
        )
        wandb.config.update(args)
        wandb.define_metric('Search_Train/Loss', summary='min')
        wandb.define_metric('Search_Train/Complexity_Loss', summary='min')
        wandb.define_metric('Search_Train/Acc', summary='max')
        wandb.define_metric('Search_Test/Loss', summary='min')
        wandb.define_metric('Search_Test/Acc', summary='max')
        wandb.define_metric('bitops-best', summary='last')
        wandb.define_metric('bita-best', summary='last')
        wandb.define_metric('bitw-best', summary='last')

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

    global ngpus_per_node
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
    if 'imagenet' in args.dataset:
        num_classes = 1000

        traindir = args.data.parent.parent / 'imagenet' / 'train'
        valdir = args.data.parent.parent / 'imagenet' / 'val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if 'inception' in args.arch:
            crop_size, short_size = 299, 342
        else:
            crop_size, short_size = 224, 256
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(short_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif 'cifar' in args.dataset:
        num_classes = 10

        # Data prep following https://github.com/kuangliu/pytorch-cifar
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Original data prep
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        #transform = transforms.Compose([
        #        #transforms.Resize((224,224)),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(),
        #        normalize,
        #    ])

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        trainset = torchvision.datasets.CIFAR10(root=args.data.parent.parent / 'cifar10', train=True,
                                                download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root=args.data.parent.parent / 'cifar10/', train=False,
                                               download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes, 
                                        reg_target=args.regularization_target,
                                        alpha_init=args.alpha_init)

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

    # group model/architecture parameters
    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]

    optimizer = torch.optim.SGD(params, args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arch_optimizer = torch.optim.SGD(alpha_params, args.lra, momentum=args.momentum,
                               weight_decay=args.alpha_decay)
    #arch_optimizer = torch.optim.Adam(alpha_params, args.lra, weight_decay=args.weight_decay)
    
    # Alternative scheduling from https://github.com/kuangliu/pytorch-cifar
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
            arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print('========= initial architecture =========')
    if hasattr(model, 'module'):
        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.module.fetch_best_arch()
    else:
        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.fetch_best_arch()
    print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
        bitops, bita, bitw))
    print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
        mixbitops, mixbita, mixbitw))
    for key, value in best_arch.items():
        print('{}: {}'.format(key, value))

    # If warmup is enabled check if pretrained model exists
    if args.warmup != 0:
        warmup_pretrained_checkpoint = args.data.parent / ('warmup_' + str(args.warmup) + '.pth.tar')
        if warmup_pretrained_checkpoint.exists():
            print(f"=> loading pretrained model '{warmup_pretrained_checkpoint}'")
            checkpoint = torch.load(warmup_pretrained_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        else:
            print(f"=> no pretrained model found at '{warmup_pretrained_checkpoint}'")
            print(f"=> warmup model for '{args.warmup}' epochs")
            # Freeze alpha parameters
            for name, param in model.named_parameters():
                if 'alpha' in name:
                    param.requires_grad = False
            # Warmup model
            warmup_best_epoch, warmup_best_acc1 = train(train_loader, val_loader, model, criterion, optimizer, arch_optimizer, scheduler, args, scope='Warmup')
            
            print(f'Best Acc@1 {warmup_best_acc1} @ epoch {warmup_best_epoch}')

            # Unfreeze alpha parameters
            for name, param in model.named_parameters():
                if 'alpha' in name:
                    param.requires_grad = True
    else:
        print('=> no warmup')

    ###############################################
    ### OLD CODE LET'S KEEP IT HERE FOR A WHILE ###
    ###############################################
    #best_epoch = args.start_epoch
    #for epoch in range(args.start_epoch, args.epochs):
    #    if args.distributed:
    #        train_sampler.set_epoch(epoch)
    #    #adjust_learning_rate(optimizer, arch_optimizer, epoch, args)

    #    # Alternative scheduling from https://github.com/kuangliu/pytorch-cifar
    #    scheduler.step()

    #    # train for one epoch
    #    train_epoch(train_loader, model, criterion, optimizer, arch_optimizer, epoch, args)

    #    print('========= architecture =========')
    #    if hasattr(model, 'module'):
    #        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.module.fetch_best_arch()
    #    else:
    #        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.fetch_best_arch()
    #    print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
    #        bitops, bita, bitw))
    #    print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
    #        mixbitops, mixbita, mixbitw))
    #    for key, value in best_arch.items():
    #        print('{}: {}'.format(key, value))

    #    # evaluate on validation set
    #    acc1 = validate(val_loader, model, criterion, epoch, args)

    #    # remember best acc@1 and save checkpoint
    #    is_best = acc1 > best_acc1
    #    best_acc1 = max(acc1, best_acc1)
    #    if is_best:
    #        best_epoch = epoch

    #    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
    #            and args.rank % ngpus_per_node == 0):
    #        save_checkpoint(args.data, {
    #            'epoch': epoch + 1,
    #            'arch': args.arch,
    #            'state_dict': model.state_dict(),
    #            'best_acc1': best_acc1,
    #            'optimizer': optimizer.state_dict(),
    #            'arch_optimizer': arch_optimizer.state_dict(),
    #        }, is_best, epoch, args.step_epoch, args)

    #    # Visualization
    #    if args.visualization:
    #        wandb.log({
    #            'Epoch': epoch,
    #            'bitops-best': bitops,
    #            'bita-best': bita,
    #            'bitw-best': bitw
    #        })

    #    # Debug: plot for each layer the fraction of selected precision for each precision
    #    if args.debug:
    #        discrete_arch = sample_arch(model.state_dict())
    #        for layer, params in discrete_arch.items():
    #            name = layer.split('.')[-1]
    #            if name == 'alpha_weight':
    #                try: 
    #                    length = params.shape[0] # works with multi-precision
    #                except:
    #                    length = 1 # works with mixed-precision
    #                if length > 1:
    #                    ch = params.shape[0]
    #                    #precisions = np.unique(params)
    #                    #n = len(precisions)
    #                    # Get explored precisions for weights from args.arch 
    #                    # args.arch is a string of the form 'mixres18_w0248a248_multiprec'
    #                    precisions = list(args.arch.split('_')[1].split('a')[0][1:])
    #                    n = len(precisions)
    #                    log_dict = {'Epoch': epoch}
    #                    for prec, bit in enumerate(precisions):
    #                        #bit = 2 ** prec if prec > 0 else 0
    #                        layer_name = '.'.join(layer.split('.')[:-2])
    #                        if layer_name == 'fc':
    #                            log_dict[layer_name+'-'+bit+'bit'] = sum(params == prec) / ch
    #                            #wandb.log({layer_name+'-'+str(bit)+'bit': sum(params+1 == prec) / ch}, 
    #                            #    step=epoch)
    #                        else:
    #                            log_dict[layer_name+'-'+bit+'bit'] = sum(params == prec) / ch
    #                            #wandb.log({layer_name+'-'+str(bit)+'bit': sum(params == prec) / ch},
    #                            #    step=epoch)
    #                        #wandb.log({
    #                        #    'Epoch': epoch,
    #                        #    '.'.join(layer.split('.')[:-2])+'-'+str(prec): sum(params == prec) / ch
    #                        #})
    #                    #wandb.log({'Epoch': epoch})
    #                    wandb.log(log_dict)

    #    # Debug: bar-plot with fraction of chosen precisions for each layer at each epoch
    #    if args.debug:
    #        discrete_arch = sample_arch(model.state_dict())
    #        table_a, table_w = get_data_table(discrete_arch) 
    #        wandb_table_a = wandb.Table(
    #            data = table_a, 
    #            columns = ['precision', 'fraction'])
    #        wandb.log({'table_a': wandb_table_a})
    #        if len(table_w[0]) > 2: # Multi-Precision
    #            wandb_table_w = wandb.Table(
    #                data = table_w, 
    #                columns = ['layer', 'precision', 'fraction'])
    #        else: # Mixed-Precision
    #            wandb_table_w = wandb.Table(
    #                data = table_w, 
    #                columns = ['precision', 'fraction'])
    #        wandb.log({'table_w': wandb_table_w})

    # Search
    best_epoch, best_acc1 = train(train_loader, val_loader, model, criterion, optimizer, arch_optimizer, scheduler, args, 
                                    scope='Search', train_sampler=train_sampler)

    # Visualization: bar-plot with fraction of chosen precisions for each layer
    if args.visualization and (not args.debug):
        discrete_arch = sample_arch(model.state_dict())
        table_a, table_w = get_data_table(discrete_arch) 
        wandb_table_a = wandb.Table(
            data = table_a, 
            columns = ['precision', 'fraction'])
        wandb.log({'table_a': wandb_table_a})
        if len(table_w[0]) > 2: # Multi-Precision
            wandb_table_w = wandb.Table(
                data = table_w, 
                columns = ['layer', 'precision', 'fraction'])
        else: # Mixed-Precision
            wandb_table_w = wandb.Table(
                data = table_w, 
                columns = ['precision', 'fraction'])
        wandb.log({'table_w': wandb_table_w})

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch))


def train(train_loader, val_loader, model, criterion, optimizer, arch_optimizer, scheduler, args, scope='Search', train_sampler=None):
    best_epoch = args.start_epoch
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, arch_optimizer, epoch, args)

        # Alternative scheduling from https://github.com/kuangliu/pytorch-cifar
        scheduler.step()

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, arch_optimizer, epoch, args, scope=scope)

        print('========= architecture =========')
        if hasattr(model, 'module'):
            best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.module.fetch_best_arch()
        else:
            best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.fetch_best_arch()
        print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
            bitops, bita, bitw))
        print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
            mixbitops, mixbita, mixbitw))
        for key, value in best_arch.items():
            print('{}: {}'.format(key, value))

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, scope=scope)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(args.data, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'arch_optimizer': arch_optimizer.state_dict(),
            }, is_best, epoch, args.step_epoch, args, scope=scope)

        # Visualization
        if args.visualization and scope == 'Search':
            wandb.log({
                scope + '/Epoch': epoch,
                'bitops-best': bitops,
                'bita-best': bita,
                'bitw-best': bitw
            })

        # Debug: plot for each layer the fraction of selected precision for each precision
        if args.debug and scope == 'Search':
            discrete_arch = sample_arch(model.state_dict())
            for layer, params in discrete_arch.items():
                name = layer.split('.')[-1]
                if name == 'alpha_weight':
                    try: 
                        length = params.shape[0] # works with multi-precision
                    except:
                        length = 1 # works with mixed-precision
                    if length > 1:
                        ch = params.shape[0]
                        #precisions = np.unique(params)
                        #n = len(precisions)
                        # Get explored precisions for weights from args.arch 
                        # args.arch is a string of the form 'mixres18_w0248a248_multiprec'
                        precisions = list(args.arch.split('_')[1].split('a')[0][1:])
                        n = len(precisions)
                        log_dict = {'Epoch': epoch}
                        for prec, bit in enumerate(precisions):
                            #bit = 2 ** prec if prec > 0 else 0
                            layer_name = '.'.join(layer.split('.')[:-2])
                            if layer_name == 'fc':
                                log_dict[layer_name+'-'+bit+'bit'] = sum(params == prec) / ch
                                #wandb.log({layer_name+'-'+str(bit)+'bit': sum(params+1 == prec) / ch}, 
                                #    step=epoch)
                            else:
                                log_dict[layer_name+'-'+bit+'bit'] = sum(params == prec) / ch
                                #wandb.log({layer_name+'-'+str(bit)+'bit': sum(params == prec) / ch},
                                #    step=epoch)
                            #wandb.log({
                            #    'Epoch': epoch,
                            #    '.'.join(layer.split('.')[:-2])+'-'+str(prec): sum(params == prec) / ch
                            #})
                        #wandb.log({'Epoch': epoch})
                        wandb.log(log_dict)

        # Debug: bar-plot with fraction of chosen precisions for each layer at each epoch
        if args.debug and scope == 'Search':
            discrete_arch = sample_arch(model.state_dict())
            table_a, table_w = get_data_table(discrete_arch) 
            wandb_table_a = wandb.Table(
                data = table_a, 
                columns = ['precision', 'fraction'])
            wandb.log({'table_a': wandb_table_a})
            if len(table_w[0]) > 2: # Multi-Precision
                wandb_table_w = wandb.Table(
                    data = table_w, 
                    columns = ['layer', 'precision', 'fraction'])
            else: # Mixed-Precision
                wandb_table_w = wandb.Table(
                    data = table_w, 
                    columns = ['precision', 'fraction'])
            wandb.log({'table_w': wandb_table_w})
    return best_epoch, best_acc1

def train_epoch(train_loader, model, criterion, optimizer, arch_optimizer, epoch, args, scope='Search'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    curr_lra = arch_optimizer.param_groups[0]['lr']
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t"
               "LRA: {}\t".format(epoch, args.epochs, curr_lr, curr_lra))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # complexity penalty
        if args.complexity_decay != 0 and scope == 'Search':
            if hasattr(model, 'module'):
                loss_complexity = args.complexity_decay * model.module.complexity_loss()
            else:
                loss_complexity = args.complexity_decay * model.complexity_loss()
            loss += loss_complexity
        else:
            loss_complexity = 0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        arch_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            #for m in model.modules():
            #    if isinstance(m, model.conv_func):
            #        if not m.fc:
            #            print("{}".format(m.mix_weight.alpha_weight))
            #        if sum(m.mix_weight.alpha_weight>100)>=1:
            #        import pdb; pdb.set_trace()
            #        break
    
    # Visualization
    if args.visualization:
        wandb.log({
                scope + "_Epoch": epoch,
                scope + "_Train/Loss": losses.avg, 
                scope + "_Train/Complexity_Loss": loss_complexity, 
                scope + "_Train/Acc": top1.avg,
                scope + "_Train/lr": curr_lr,
                scope + "_Train/lra": curr_lra
            })

def validate(val_loader, model, criterion, epoch, args, scope='Search'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    
    # Visualization
    if args.visualization:
        wandb.log({
                scope + "_Epoch": epoch,
                scope + "_Test/Loss": losses.avg, 
                scope + "_Test/Acc": top1.avg
            })

    return top1.avg


def save_checkpoint(root, state, is_best, epoch, step_epoch, args, filename='arch_checkpoint.pth.tar', scope='Search'):
    if scope == 'Warmup':
        torch.save(state, root.parent / ('warmup_' + str(args.warmup) + '.pth.tar'))
    elif scope == 'Search':
        torch.save(state, root / filename)
    else:
        raise ValueError("Scope must be either 'Warmup' or 'Search'")
    if scope == 'Search':
        if is_best:
            shutil.copyfile(root / filename, root / 'arch_model_best.pth.tar')
        if (epoch + 1) % step_epoch == 0:
            shutil.copyfile(root / filename, root / 'arch_checkpoint_ep{}.pth.tar'.format(epoch + 1))


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


def adjust_learning_rate(optimizer, arch_optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step_epoch))
    lra = args.lra * (0.1 ** (epoch // args.step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in arch_optimizer.param_groups:
        param_group['lr'] = lra


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

# MR
def get_data_table(arch):
    table_a = list()
    a_buffer = list()
    table_w = list()
    w_buffer = list()
    for layer, params in arch.items():
        name = layer.split('.')[-1]
        if name == 'alpha_activ':
            a_buffer.append(params)
        elif name == 'alpha_weight':
            try: 
                length = params.shape[0] # works with multi-precision
            except:
                length = 1 # works with mixed-precision
            if length > 1:
                ch = params.shape[0]
                precisions = np.unique(params)
                for prec in precisions:
                    frac = sum(params == prec) / ch
                    table_w.append(['.'.join(layer.split('.')[:-2]), prec, frac])
            else:
                w_buffer.append(params)
    unique_prec_a = set(a_buffer)
    for prec in unique_prec_a:
        frac = sum([1 for val in a_buffer if val == prec]) / len(a_buffer)
        table_a.append([prec, frac])
    if len(w_buffer) > 0:
        unique_prec_w = set(w_buffer)
        for prec in unique_prec_w:
            frac = sum([1 for val in w_buffer if val == prec]) / len(w_buffer)
            table_w.append(['', prec, frac])
    return table_a, table_w

# MR
def sample_arch(state_dict):
    arch = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha = params.cpu().numpy()
            arch[full_name] = alpha.argmax()
        elif name == 'alpha_weight':
            alpha = params.cpu().numpy()
            arch[full_name] = alpha.argmax(axis=0)
    return arch


if __name__ == '__main__':
    main()
