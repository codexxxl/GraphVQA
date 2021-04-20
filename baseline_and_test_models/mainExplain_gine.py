"""

File Name: main.py

Single GPU training: 
CUDA_VISIBLE_DEVICES=2 python mainExplain.py --log-name debug.log && echo 'Ground Truth Scene Graph Debug'




Distributed Training:
CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env mainExplain.py

Kill Distributed:
kill $(ps aux | grep mainExplain.py | grep -v grep | awk '{print $2}')


Do Evaluation:
# Calculate model accuracy (program, short, and full) on val set

CUDA_VISIBLE_DEVICES=3 python mainExplain.py \
    --evaluate \
    --resume ./gtsg_large_cap_outputdir/checkpoint0029.pth && echo 'test dump '


on testdev set
CUDA_VISIBLE_DEVICES=2 python /home/ubuntu/GQA/DialogGQA/GraphVQA-master/mainExplain.py \
    --evaluate \
    --evaluate_sets testdev \
    --resume ./gtsg_large_cap_outputdir/checkpoint0029.pth

"""
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import logging
from tqdm import tqdm
import torch
import torch_geometric
import torch.backends.cudnn as cudnn
import pathlib
import util.misc as utils

from gqa_dataset_entry import GQATorchDataset, GQATorchDataset_collate_fn
from pipeline_model_gine import PipelineModel # use naive GINE model
import json
# GPU settings
# assert torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
# device = torch.device("cuda")
# torch.backends.cudnn.benchmark = True
# cudnn.benchmark = True
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Default CUDA device

def get_args_parser():
    parser = argparse.ArgumentParser('Explainable GQA Parser', add_help=False)
    parser.add_argument('--data', metavar='PATH', default='./',
                        help='path to dataset')
    parser.add_argument('--save-dir', metavar='PATH', default='./',
                        help='path to dataset')
    # parser.add_argument('--log-name', default='tmp.log', type=str, metavar='PATH',
    # parser.add_argument('--log-name', default='detrDEBUG.log', type=str, metavar='PATH',
    # parser.add_argument('--log-name', default='detr.log', type=str, metavar='PATH',
    # parser.add_argument('--log-name', default='detrDEV.log', type=str, metavar='PATH',
    parser.add_argument('--log-name', default='gtsg.log', type=str, metavar='PATH',
                        help='path to the log file (default: output.log)')
    # parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    # parser.add_argument('-b', '--batch-size', default=1024, type=int,
    # parser.add_argument('-b', '--batch-size', default=512, type=int,
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # parser.add_argument('-p', '--print-freq', default=1, type=int,
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--evaluate_sets', default=['val_unbiased'], nargs='+',
                        help='Data sets/splits to perform evaluation, e.g. '
                             'val_unbiased, testdev etc. Multiple sets/splits '
                             'are supported and need to be separated by space')
    # parser.add_argument('--seed', default=1234, type=int,
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                         'N processes per node, which has N GPUs. This is the '
    #                         'fastest way to use PyTorch for either single node or '
    #                         'multi node data parallel training')

    parser.add_argument('--output_dir', default='./outputdir',
                        help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


"""
Seems equivalent to
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
"""
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr * (0.1 ** (epoch // 15)) # experimenting to speed up convergence
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# def main():
#     args = parser.parse_args()

def main(args):
    # args = parser.parse_args()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    if args.seed is not None:
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    ##################################
    # Logging setting
    ##################################
    if args.output_dir and utils.is_main_process():
        logging.basicConfig(
            filename=os.path.join(args.output_dir, args.log_name),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Save to logging
    ##################################
    if utils.is_main_process():
        logging.info(str(args))

    ##################################
    # Initialize dataset
    ##################################

    if not args.evaluate:
        # build_vocab_flag=True, # Takes a long time to build a vocab
        train_dataset = GQATorchDataset(
            split='train_unbiased',
            build_vocab_flag=False,
            load_vocab_flag=False
        )

        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(train_dataset)
        else:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler_train,
            collate_fn=GQATorchDataset_collate_fn,
            num_workers=args.workers
        )

        # Old version
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, shuffle=True,
        #     collate_fn=GQATorchDataset_collate_fn,
        #     num_workers=args.workers, pin_memory=True)

    val_dataset_list = []
    for eval_split in args.evaluate_sets:
        val_dataset_list.append(GQATorchDataset(
            split=eval_split,
            build_vocab_flag=False,
            load_vocab_flag=args.evaluate
        ))
    val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)

    if args.distributed:
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=GQATorchDataset_collate_fn,
        num_workers=args.workers
    )

    # Old version
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     collate_fn=GQATorchDataset_collate_fn,
    #     num_workers=args.workers, pin_memory=True)

    ##################################
    # Initialize model
    # - note: must init dataset first. Since we will use the vocab from the dataset
    ##################################
    model = PipelineModel()

    ##################################
    # Deploy model on GPU
    ##################################
    model = model.to(device=cuda)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    ##################################
    # define optimizer (and scheduler)
    ##################################

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0, #  weight_decay=args.weight_decay
        amsgrad=False,
    )
    # optimizer = torch.optim.AdamW(
    #     params=model.parameters(),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.evaluate:
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'lr_scheduler' in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if 'epoch' in checkpoint:
                    args.start_epoch = checkpoint['epoch'] + 1

            # checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    ##################################
    # Define loss functions (criterion)
    ##################################
    # criterion = torch.nn.CrossEntropyLoss().cuda()

    text_pad_idx = GQATorchDataset.TEXT.vocab.stoi[GQATorchDataset.TEXT.pad_token]
    criterion = {
        "program": torch.nn.CrossEntropyLoss(ignore_index=text_pad_idx).to(device=cuda),
        "full_answer": torch.nn.CrossEntropyLoss(ignore_index=text_pad_idx).to(device=cuda),
        "short_answer": torch.nn.CrossEntropyLoss().to(device=cuda),
        # "short_answer": torch.nn.BCEWithLogitsLoss().to(device=cuda), # sigmoid
        "execution_bitmap": torch.nn.BCELoss().to(device=cuda),
    }

    ##################################
    # If Evaluate Only
    ##################################

    if args.evaluate:
        validate(val_loader, model, criterion, args, DUMP_RESULT=True)
        return

    ##################################
    # Main Training Loop
    ##################################

    # best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            ##################################
            # In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method
            # at the beginning of each epoch before creating the DataLoader iterator is necessary
            # to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.
            ##################################
            sampler_train.set_epoch(epoch)

        lr_scheduler.step()

        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        if (epoch + 1) % 5 == 0:
            validate(val_loader, model, criterion, args, FAST_VALIDATE_FLAG=False)

        # # remember best acc@1 and save checkpoint
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     # 'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     # 'best_acc1': best_acc1,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best)

        if args.output_dir:
            output_dir = pathlib.Path(args.output_dir)
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    losses = AverageMeter('Loss', ':.2e')

    program_acc = AverageMeter('Acc@Program', ':4.2f')
    program_group_acc = AverageMeter('Acc@ProgramGroup', ':4.2f')
    program_non_empty_acc = AverageMeter('Acc@ProgramNonEmpty', ':4.2f')

    # bitmap_precision = AverageMeter('Precision@Bitmap', ':4.2f')
    # bitmap_recall = AverageMeter('Recall@Bitmap', ':4.2f')

    # full_answer_acc = AverageMeter('Acc@Full', ':4.2f')
    short_answer_acc = AverageMeter('Acc@Short', ':4.2f')

    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, losses,
            program_acc, program_group_acc, program_non_empty_acc,
            short_answer_acc
        ],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data_batch) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        questionID, questions, gt_scene_graphs, programs, full_answers, short_answer_label, types = data_batch
        del questionID
        questions, gt_scene_graphs, programs, full_answers, short_answer_label = [
            datum.to(device=cuda, non_blocking=True) for datum in [
                questions, gt_scene_graphs, programs, full_answers, short_answer_label
            ]
        ]

        this_batch_size = questions.size(1)
        # print("this_batch_size", this_batch_size, "data_batch", data_batch)

        ##################################
        # Prepare training input and training target for text generation
        # - shape [len, batch]
        ##################################
        programs_input = programs[:-1]
        programs_target = programs[1:]
        full_answers_input = full_answers[:-1]
        full_answers_target = full_answers[1:]
        # print("programs_input.size()", programs_input.size())
        # print("programs_target.size()", programs_target.size())
        # print("full_answers_input.size()", full_answers_input.size())
        # print("full_answers_target.size()", full_answers_target.size())

        # print("programs_input", programs_input)
        # print("programs_target", programs_target)
        # print("full_answers_input", full_answers_input)
        # print("full_answers_target", full_answers_target)

        ##################################
        # Forward training data
        ##################################
        output = model(
            questions,
            gt_scene_graphs,
            programs_input,
            full_answers_input
        )
        programs_output, short_answer_logits = output

        ##################################
        # Evaluate on training data
        ##################################
        with torch.no_grad():
            ##################################
            # Calculate Fast Evaluation for each module
            ##################################
            this_short_answer_acc1 = accuracy(short_answer_logits, short_answer_label, topk=(1,))
            short_answer_acc.update(this_short_answer_acc1[0].item(), this_batch_size)

            text_pad_idx = GQATorchDataset.TEXT.vocab.stoi[GQATorchDataset.TEXT.pad_token]

            ##################################
            # Convert output probability to top1 guess
            # So that we could measure accuracy
            ##################################
            programs_output_pred = programs_output.detach().topk(
                k=1, dim=-1, largest=True, sorted=True
            )[1].squeeze(-1)
            # full_answers_output_pred = full_answers_output.detach().topk(
            #     k=1, dim=-1, largest=True, sorted=True
            # )[1].squeeze(-1)

            this_program_acc, this_program_group_acc, this_program_non_empty_acc = program_string_exact_match_acc(
                programs_output_pred, programs_target,
                padding_idx=text_pad_idx,
                group_accuracy_WAY_NUM=GQATorchDataset.MAX_EXECUTION_STEP)
            program_acc.update(this_program_acc, this_batch_size)
            program_group_acc.update(this_program_group_acc, this_batch_size // GQATorchDataset.MAX_EXECUTION_STEP)
            program_non_empty_acc.update(this_program_non_empty_acc, this_batch_size)

            # this_full_answers_acc = string_exact_match_acc(
            #     full_answers_output_pred, full_answers_target, padding_idx=text_pad_idx
            # )
            # full_answer_acc.update(this_full_answers_acc, this_batch_size)

        ##################################
        # Neural Execution Engine Bitmap loss
        # ground truth stored at gt_scene_graphs.y
        # using torch.nn.BCELoss - torch.nn.functional.binary_cross_entropy
        # should also add a special precision recall for that
        ##################################
        # execution_bitmap_loss = criterion['execution_bitmap'](execution_bitmap, gt_scene_graphs.y)

        # precision, precision_div, recall, recall_div = bitmap_precision_recall(
        #     execution_bitmap, gt_scene_graphs.y, threshold=0.5
        # )

        # bitmap_precision.update(precision, precision_div)
        # bitmap_recall.update(recall, recall_div)

        ##################################
        # Calculate each module's loss and get global loss
        ##################################
        def text_generation_loss(loss_fn, output, target):
            text_vocab_size = len(GQATorchDataset.TEXT.vocab)
            output = output.contiguous().view(-1, text_vocab_size)
            target = target.contiguous().view(-1)
            this_text_loss = loss_fn(output, target)
            return this_text_loss

        program_loss = text_generation_loss(criterion['program'], programs_output, programs_target)
        # full_answer_loss = text_generation_loss(
        #     criterion['full_answer'], full_answers_output, full_answers_target
        # )

        ##################################
        # using sigmoid loss for short answer
        ##################################
        # num_short_answer_choices = 1842
        # short_answer_label_one_hot = torch.nn.functional.one_hot(short_answer_label, num_short_answer_choices).float()
        # short_answer_loss = criterion['short_answer'](short_answer_logits, short_answer_label_one_hot) # sigmoid loss

        ##################################
        # normal softmax loss for short answer
        ##################################
        short_answer_loss = criterion['short_answer'](short_answer_logits, short_answer_label)
        # loss = program_loss + full_answer_loss + short_answer_loss # + execution_bitmap_loss
        loss = program_loss +  short_answer_loss 
        # measure accuracy and record loss
        losses.update(loss.item(), this_batch_size)

        ##################################
        # compute gradient and do SGD step
        ##################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##################################
        # measure elapsed time
        ##################################
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            progress.display(i)

    ##################################
    # Give final score
    ##################################
    progress.display(batch=len(train_loader))


"""
Input shape: [Len, Batch]
A fast GPU-based string exact match accuracy calculator

TODO: if the prediction does not stop at target's padding area.
(should rarely happen if at all)
"""
def string_exact_match_acc(predictions, target, padding_idx=1):

    ##################################
    # Do token-level match first
    # Generate a matching matrix: if equals or pad, then put 1, else 0
    # Shape: [Len, Batch]
    ##################################
    target_len = target.size(0)
    # truncated
    predictions = predictions[:target_len]
    char_match_matrix = (predictions == target).long()
    # print("char_match_matrix", char_match_matrix)
    cond_match_matrix = torch.where(target == padding_idx, torch.ones_like(target), char_match_matrix)
    # print("cond_match_matrix", cond_match_matrix)
    del char_match_matrix

    ##################################
    # Reduction of token-level match
    # 1 means exact match, 0 means at least one token not matching
    # Dim: note that the first dim is len, batch is the second dim
    ##################################
    # ret: (values, indices)
    match_reduced, _ = torch.min(input=cond_match_matrix, dim=0, keepdim=False)
    # print("match_reduced", match_reduced)
    this_batch_size = target.size(1)
    # print("this_batch_size", this_batch_size)
    # mul 100, converting to percentage
    accuracy = torch.sum(match_reduced).item() / this_batch_size * 100.0

    return accuracy


"""
Input shape: [Len, Batch]
A fast GPU-based string exact match accuracy calculator

TODO: if the prediction does not stop at target's padding area.
(should rarely happen if at all)

group_accuracy_WAY_NUM: only calculated as correct if the whole group is correct.
Used in program accuracy: only correct if all instructions are correct.
-1 means ignore
"""
def program_string_exact_match_acc(predictions, target, padding_idx=1, group_accuracy_WAY_NUM=-1):

    ##################################
    # Do token-level match first
    # Generate a matching matrix: if equals or pad, then put 1, else 0
    # Shape: [Len, Batch]
    ##################################
    target_len = target.size(0)
    # truncated
    predictions = predictions[:target_len]
    char_match_matrix = (predictions == target).long()
    cond_match_matrix = torch.where(target == padding_idx, torch.ones_like(target), char_match_matrix)
    del char_match_matrix

    ##################################
    # Reduction of token-level match
    # 1 means exact match, 0 means at least one token not matching
    # Dim: note that the first dim is len, batch is the second dim
    ##################################
    # ret: (values, indices)
    match_reduced, _ = torch.min(input=cond_match_matrix, dim=0, keepdim=False)
    this_batch_size = target.size(1)
    # mul 100, converting to percentage
    accuracy = torch.sum(match_reduced).item() / this_batch_size * 100.0

    ##################################
    # Calculate Batch Accuracy
    ##################################
    group_batch_size = this_batch_size // group_accuracy_WAY_NUM
    match_reduced_group_reshape = match_reduced.view(group_batch_size, group_accuracy_WAY_NUM)
    # print("match_reduced_group_reshape", match_reduced_group_reshape)
    # ret: (values, indices)
    group_match_reduced, _ = torch.min(input=match_reduced_group_reshape, dim=1, keepdim=False)
    # print("group_match_reduced", group_match_reduced)
    # mul 100, converting to percentage
    group_accuracy = torch.sum(group_match_reduced).item() / group_batch_size * 100.0

    ##################################
    # Calculate Empty
    # start of sentence, end of sentence, padding
    # Shape: [Len=2, Batch]
    ##################################
    # empty and counted as correct
    empty_instr_flag = (target[2] == padding_idx) & match_reduced.bool()
    empty_instr_flag = empty_instr_flag.long()
    # print("empty_instr_flag", empty_instr_flag)
    empty_count = torch.sum(empty_instr_flag).item()
    # print("empty_count", empty_count)
    non_empty_accuracy = (torch.sum(match_reduced).item() - empty_count) / (this_batch_size - empty_count) * 100.0

    ##################################
    # Return
    ##################################
    return accuracy, group_accuracy , non_empty_accuracy


def validate(val_loader, model, criterion, args, FAST_VALIDATE_FLAG=False, DUMP_RESULT=False):
    batch_time = AverageMeter('Time', ':6.3f')

    program_acc = AverageMeter('Acc@Program', ':6.2f')
    program_group_acc = AverageMeter('Acc@ProgramGroup', ':4.2f')
    program_non_empty_acc = AverageMeter('Acc@ProgramNonEmpty', ':4.2f')

    # bitmap_precision = AverageMeter('Precision@Bitmap', ':4.2f')
    # bitmap_recall = AverageMeter('Recall@Bitmap', ':4.2f')

    # full_answer_acc = AverageMeter('Acc@Full', ':6.2f')
    short_answer_acc = AverageMeter('Acc@Short', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [
            batch_time, program_acc,
            program_group_acc, program_non_empty_acc,
            short_answer_acc
        ],
        prefix='Test: '
    )

    # switch to evaluate mode
    model.eval()

    if DUMP_RESULT:
        quesid2ans = {}

    with torch.no_grad():
        end = time.time()
        for i, (data_batch) in enumerate(val_loader):

            questionID, questions, gt_scene_graphs, programs, full_answers, short_answer_label, types = data_batch

            questions, gt_scene_graphs, programs, full_answers, short_answer_label = [
                datum.to(device=cuda, non_blocking=True) for datum in [
                    questions, gt_scene_graphs, programs, full_answers, short_answer_label
                ]
            ]

            this_batch_size = questions.size(1)

            if FAST_VALIDATE_FLAG:
                raise NotImplementedError("Should not use fast validation. Only for short answer accuracy")
                ##################################
                # Prepare training input and training target for text generation
                ##################################
                programs_input = programs[:-1]
                programs_target = programs[1:]
                full_answers_input = full_answers[:-1]
                full_answers_target = full_answers[1:]

                ##################################
                # Forward evaluate data
                ##################################
                output = model(
                    questions,
                    gt_scene_graphs,
                    programs_input,
                    full_answers_input
                )
                programs_output, short_answer_logits = output

                ##################################
                # Convert output probability to top1 guess
                # So that we could measure accuracy
                ##################################
                programs_output_pred = programs_output.detach().topk(
                    k=1, dim=-1, largest=True, sorted=True
                )[1].squeeze(-1)
                # full_answers_output_pred = full_answers_output.detach().topk(
                #     k=1, dim=-1, largest=True, sorted=True
                # )[1].squeeze(-1)

            else:

                programs_target = programs
                full_answers_target = full_answers

                ##################################
                # Greedy decoding-based evaluation
                ##################################
                output = model(
                    questions,
                    gt_scene_graphs,
                    None,
                    None,
                    SAMPLE_FLAG=True
                )
                programs_output_pred,  short_answer_logits = output

            ##################################
            # Neural Execution Engine Bitmap loss
            # ground truth stored at gt_scene_graphs.y
            # using torch.nn.BCELoss - torch.nn.functional.binary_cross_entropy
            ##################################
            # precision, precision_div, recall, recall_div = bitmap_precision_recall(
            #     execution_bitmap, gt_scene_graphs.y, threshold=0.5
            # )

            # bitmap_precision.update(precision, precision_div)
            # bitmap_recall.update(recall, recall_div)

            ##################################
            # Calculate Fast Evaluation for each module
            ##################################
            this_short_answer_acc1 = accuracy(short_answer_logits.detach(), short_answer_label, topk=(1,))
            short_answer_acc.update(this_short_answer_acc1[0].item(), this_batch_size)

            text_pad_idx = GQATorchDataset.TEXT.vocab.stoi[GQATorchDataset.TEXT.pad_token]
            this_program_acc, this_program_group_acc, this_program_non_empty_acc = program_string_exact_match_acc(
                programs_output_pred, programs_target,
                padding_idx=text_pad_idx,
                group_accuracy_WAY_NUM=GQATorchDataset.MAX_EXECUTION_STEP
            )
            program_acc.update(this_program_acc, this_batch_size)
            program_group_acc.update(this_program_group_acc, this_batch_size // GQATorchDataset.MAX_EXECUTION_STEP)
            program_non_empty_acc.update(this_program_non_empty_acc, this_batch_size)

            # this_full_answers_acc = string_exact_match_acc(
            #     full_answers_output_pred.detach(), full_answers_target, padding_idx=text_pad_idx
            # )
            # full_answer_acc.update(this_full_answers_acc, this_batch_size)

            ##################################
            # Example Visualization from the first batch
            ##################################

            if i == 0 and True:
                for batch_idx in range(min(this_batch_size, 128)):

                    ##################################
                    # print Question and Question ID
                    ##################################
                    question = questions[:, batch_idx].cpu()
                    question_sent, _ = GQATorchDataset.indices_to_string(question, True)
                    print("Question({}) QID({}):".format(batch_idx, questionID[batch_idx]), question_sent)
                    if utils.is_main_process():
                        logging.info("Question({}) QID({}): {}".format(batch_idx, questionID[batch_idx], question_sent))

                    ##################################
                    # print program prediction
                    ##################################

                    for instr_idx in range(GQATorchDataset.MAX_EXECUTION_STEP):
                        true_batch_idx = instr_idx + GQATorchDataset.MAX_EXECUTION_STEP * batch_idx
                        gt = programs[:, true_batch_idx].cpu()
                        pred = programs_output_pred[:, true_batch_idx]
                        pred_sent, _ = GQATorchDataset.indices_to_string(pred, True)
                        gt_sent, _ = GQATorchDataset.indices_to_string(gt, True)

                        if len(pred_sent) == 0 and len(gt_sent) == 0:
                            # skip if both target and prediciton are empty
                            continue

                        # gt_caption
                        print(
                            "Generated Program ({}): ".format(true_batch_idx), pred_sent,
                            " Ground Truth Program ({}):".format(true_batch_idx), gt_sent
                        )
                        if utils.is_main_process():
                            # gt_caption
                            logging.info("Generated Program ({}): {}  Ground Truth Program ({}): {}".format(
                                true_batch_idx, pred_sent, true_batch_idx, gt_sent
                            ))

                    ##################################
                    # print full answer prediction
                    ##################################
                    # gt = full_answers[:, batch_idx].cpu()
                    # pred = full_answers_output_pred[:, batch_idx]
                    # pred_sent, _ = GQATorchDataset.indices_to_string(pred, True)
                    # gt_sent, _ = GQATorchDataset.indices_to_string(gt, True)
                    # # gt_caption
                    # print(
                    #     "Generated Full Answer ({}): ".format(batch_idx), pred_sent,
                    #     "Ground Truth Full Answer ({}):".format(batch_idx), gt_sent
                    # )
                    # if utils.is_main_process():
                    #     # gt_caption
                    #     logging.info("Generated Full Answer ({}): {} Ground Truth Full Answer ({}): {}".format(
                    #         batch_idx, pred_sent, batch_idx, gt_sent
                    #     ))

            ##################################
            # Dump Results if enabled
            ##################################
            if DUMP_RESULT:

                short_answer_pred_score, short_answer_pred_label = short_answer_logits.max(1)
                short_answer_pred_score, short_answer_pred_label = short_answer_pred_score.cpu(), short_answer_pred_label.cpu()
                for batch_idx in range( this_batch_size ):
                    ##################################
                    # print Question and Question ID
                    ##################################
                    question = questions[:, batch_idx].cpu()
                    question_sent, _ = GQATorchDataset.indices_to_string(question, True)

                    ##################################
                    # print program prediction
                    ##################################
                    ground_truth_program_list = []
                    predicted_program_list = []
                    for instr_idx in range(GQATorchDataset.MAX_EXECUTION_STEP):
                        true_batch_idx = instr_idx + GQATorchDataset.MAX_EXECUTION_STEP * batch_idx
                        gt = programs[:, true_batch_idx].cpu()
                        pred = programs_output_pred[:, true_batch_idx]
                        pred_sent, _ = GQATorchDataset.indices_to_string(pred, True)
                        gt_sent, _ = GQATorchDataset.indices_to_string(gt, True)

                        if len(pred_sent) == 0 and len(gt_sent) == 0:
                            # skip if both target and prediciton are empty
                            continue

                        ground_truth_program_list.append(gt_sent)
                        predicted_program_list.append(pred_sent)


                    ##################################
                    # print full answer prediction
                    ##################################
                    # gt = full_answers[:, batch_idx].cpu()
                    # pred = full_answers_output_pred[:, batch_idx]
                    # pred_sent, _ = GQATorchDataset.indices_to_string(pred, True)
                    # gt_sent, _ = GQATorchDataset.indices_to_string(gt, True)
                    # gt_caption

                    ##################################
                    # get short answer prediction
                    ##################################
                    qid = questionID[batch_idx]
                    quesid2ans[qid] = {
                        "questionId": str(qid),
                        "question": question_sent,
                        "ground_truth_program_list": ground_truth_program_list,
                        "predicted_program_list": predicted_program_list,
                        "answer": GQATorchDataset.label2ans[short_answer_label[batch_idx].cpu().item()],
                        # predicted short answer
                        "prediction": GQATorchDataset.label2ans[short_answer_pred_label[batch_idx].cpu().item()],
                        "prediction_score": '{:.2f}'.format(short_answer_pred_score[batch_idx].cpu().item()),
                        "types": types[batch_idx],
                    }

            ##################################
            # measure elapsed time
            ##################################
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                progress.display(i)

            ##################################
            # Only for dubugging: short cut the evaluation loop
            ##################################
            # break

    ##################################
    # Give final score
    ##################################
    progress.display(batch=len(val_loader))

    if DUMP_RESULT:
        result_dump_path = os.path.join(args.output_dir, "dump_results.json")
        with open(result_dump_path, 'w') as f:
            json.dump(quesid2ans, f, indent=4, sort_keys=True)
            print("Result Dumped!", str(result_dump_path))


    return


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

        ##################################
        # Save to logging
        ##################################
        if utils.is_main_process():
            logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def bitmap_precision_recall(output, target, threshold=0.5):
    """ Computes the precision recall over execution bitmap given a interpretation threshold """
    with torch.no_grad():

        target_one = (target == 1)
        # target_one_total = torch.sum(target_one).item()

        output_pred_binary = (output > threshold)

        true_positive = (output_pred_binary & target_one)
        false_negative = (torch.logical_not(output_pred_binary) & target_one)
        false_positive = (output_pred_binary & torch.logical_not(target_one))
        # true_negative = ( torch.logical_not(output_pred_binary) & torch.logical_not(target_one))

        tp = true_positive.float().sum().item()
        fn = false_negative.float().sum().item()
        fp = false_positive.float().sum().item()
        # tn = true_negative.float().sum().item()

        precision_div = (tp + fp)
        precision = tp / precision_div * 100. if precision_div != 0 else 0.
        # prevent div 0
        precision_div = 1e-6 if precision_div == 0 else precision_div

        recall_div = (tp + fn)
        recall = tp / recall_div * 100. if recall_div != 0 else 0.
        # prevent div 0
        recall_div = 1e-6 if recall_div == 0 else recall_div

        return precision, precision_div, recall, recall_div


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainable GQA training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    ##################################
    # Initialize saving directory
    ##################################
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
