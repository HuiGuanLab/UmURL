import argparse
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set


import torch.distributed as dist
from distributed import init_distributed_mode

from umurl import UmURL


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)


parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='traiining protocol cross_view/cross_subject/cross_setup')


parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def main():
    args = parser.parse_args()
    init_distributed_mode(args)
    gpu = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    

    # pretraining dataset and protocol
    from options import options_pretraining as options 
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()

    # create model
    print("=> creating model")

    model = UmURL(**opts.encoder_args)
    print("options",opts.train_feeder_args)
    print(model)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
   
    model = model.cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        


    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
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
                #checkpoint = torch.load(args.resume, map_location=loc)
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    ## Data loading code
    train_dataset = get_pretraining_set(opts)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    if args.rank==0:
        writer = SummaryWriter(args.checkpoint_path)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        loss = train(gpu, scaler, train_loader, model, criterion, optimizer, epoch, args)
        if args.rank==0:
            writer.add_scalar('train_loss', loss.avg, global_step=epoch)

        if epoch % 50 == 0 and args.rank==0:
                  save_checkpoint({
                      'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                  }, is_best=False, filename=args.checkpoint_path+'/checkpoint_{:04d}.pth.tar'.format(epoch,loss.avg))

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vc_reg(x):
    B, D = x.shape

    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_loss = torch.mean(torch.relu(1 - std_x))

    z = x - x.mean(dim=0)
    cov_z = (z.T @ z) / (B - 1) 
    cov_loss = off_diagonal(cov_z).pow(2).sum() / D

    return 5 * std_loss + cov_loss


def train(gpu, scaler, train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses,],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch,optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()


    end = time.time()
    for i, (data_v1, data_v2, data_v3, data_v4) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            data_v1 = data_v1.float().cuda(gpu, non_blocking=True)
            data_v2 = data_v2.float().cuda(gpu, non_blocking=True)
            data_v3 = data_v3.float().cuda(gpu, non_blocking=True)
            data_v4 = data_v4.float().cuda(gpu, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # compute output
            z_j, z_b, z_m, z_uj, z_ub, z_um = model(data_v1, data_v2, data_v3, data_v4)

            z_j = torch.cat(FullGatherLayer.apply(z_j), dim=0)
            z_b = torch.cat(FullGatherLayer.apply(z_b), dim=0)
            z_m = torch.cat(FullGatherLayer.apply(z_m), dim=0)

            z_uj = torch.cat(FullGatherLayer.apply(z_uj), dim=0)
            z_ub = torch.cat(FullGatherLayer.apply(z_ub), dim=0)
            z_um = torch.cat(FullGatherLayer.apply(z_um), dim=0)


            B, D = z_j.shape
            
            # intra-modal consistency
            intra = criterion(z_j, z_uj) + criterion(z_b, z_ub) + criterion(z_m, z_um)

            # inter-modal consistency  
            center = (z_j + z_b + z_m) / 3   
            inter = criterion(z_j, center) + criterion(z_b, center) + criterion(z_m, center) 

            # vc regularization
            reg = vc_reg(z_j) + vc_reg(z_b) + vc_reg(z_m) \
                + vc_reg(z_uj) + vc_reg(z_ub) + vc_reg(z_um)   
            
            loss = 5 * ( intra + inter) + reg

        losses.update(loss.item(), B)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
       
    return losses

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        print('\t'.join(entries),flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def linear_warmup(optimizer, step, args):
    lr = args.lr
    learning_rate = (step / float(args.warmup_steps) * lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / (args.epochs - args.warmup_epochs)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
