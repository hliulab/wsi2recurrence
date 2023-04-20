import builtins
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import time
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from Adco.model.AdCo import AdCo, Adversary_Negatives
from Adco.ops.os_operation import mkdir
from Adco.training.train_utils import adjust_learning_rate,save_checkpoint
from Adco.training.train import train, init_memory
from Adco.data_processing.loader import TwoCropsTransform, GaussianBlur
import pandas as pd

from datasets.dataset_nolabel_folder import FolderDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 0'

def init_log_path(args):
    """
    :param args:
    :return:
    save model+log path
    """
    save_path = os.path.join(args.save_path, args.log_path)
    if args.gpu == 0:
        mkdir(save_path)
    save_path = os.path.join(save_path, args.dataset)
    if args.gpu == 0:
        mkdir(save_path)
    save_path = os.path.join(save_path, "lr_" + str(args.lr) + "_memlr" + str(args.memory_lr))
    if args.gpu == 0:
        mkdir(save_path)
    save_path = os.path.join(save_path, "cos_" + str(args.cos))
    if args.gpu == 0:
        mkdir(save_path)
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    save_path = os.path.join(save_path, formatted_today + now)
    if args.gpu == 0:
        mkdir(save_path)
    return save_path

def main_worker(gpu, ngpus_per_node, args):
    params = vars(args)
    args.gpu = gpu
    print(vars(args))

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # args.device = device
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
    
        builtins.print = print_pass

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

    print("=> creating model '{}'".format(args.arch))
    multi_crop = args.multi_crop
    Memory_Bank = Adversary_Negatives(args.cluster, args.moco_dim, multi_crop)

    model = AdCo(args, args.moco_dim, args.moco_m, args.moco_t, args.mlp)
    
    print(args.distributed)
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
            Memory_Bank = Memory_Bank.cuda(args.gpu)
        else:
            model.cuda()
            Memory_Bank.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Memory_Bank=Memory_Bank.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    #Memory_Bank = DataParallel(Memory_Bank)
    #model = DataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Memory_Bank.load_state_dict(checkpoint['Memory_Bank'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True
    # 加载数据

    # 生成train_loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.multi_crop:
        from Adco.data_processing.MultiCrop_Transform import Multi_Transform
        multi_transform = Multi_Transform(args.size_crops,
                                          args.nmb_crops,
                                          args.min_scale_crops,
                                          args.max_scale_crops, normalize)
        train_dataset = FolderDataset(
            args.data, multi_transform)
    else:
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        train_dataset = FolderDataset(args.data,args.data1,TwoCropsTransform(transforms.Compose(augmentation)))
    total_length = train_dataset.length
    print(f"一共有{total_length}张patch")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    save_path = init_log_path(args)
    bank_size=args.cluster
    print(f"Memory Bank的类型:{type(Memory_Bank.W)}")
    print(f"init memory前，Memory Bank的形状{Memory_Bank.W.shape}")
    model.eval()
    #init memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        init_memory(train_loader, model, Memory_Bank, criterion,
              optimizer, 0,args)
        print("Init memory bank finished!!")
    print(f"init memory后，Memory Bank的形状{Memory_Bank.W.shape}")
    best_Acc=0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1 = train(train_loader, model, Memory_Bank, criterion,
                     optimizer, epoch, args)
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print(args.multiprocessing_distributed)
            print(args.rank % ngpus_per_node == 0)
            save_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'best_acc': best_Acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'Memory_Bank': Memory_Bank.state_dict(),
            }
    
            if epoch % 10 == 9:
                print(epoch)
                tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')
    
            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
        # # if epoch%10==9:
        # #     save_dict={
        # #     'epoch': epoch + 1,
        # #     'best_acc':best_Acc,
        # #     'state_dict': model.state_dict(),
        # #     # 'optimizer': optimizer.state_dict(),
        # #     # 'Memory_Bank':Memory_Bank.state_dict(),
        # #     }
        # #
        # #     tmp_save_path = os.path.join(args.save_path, 'checkpoint_{}_not_sym_128_{:04d}.pth.tar'.format(args.data_type, epoch))
        # #     save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
        # save_dict = {
        #     'arch': args.arch,
        #     'best_acc': best_Acc,
        #     'state_dict': model.state_dict(),
        # # 'optimizer': optimizer.state_dict(),
        # # 'Memory_Bank': Memory_Bank.state_dict(),
        # }
        # if epoch % 10 == 9:
        #     tmp_save_path = os.path.join(args.save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
        #     save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
        # tmp_save_path = os.path.join(args.save_path, f'cl_10_{args.data_type}_best.pth.tar')
        # save_checkpoint(save_dict,is_best=True, filename=tmp_save_path)
