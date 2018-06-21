import argparse
import math
import os
import random

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import PTBLoader
from helpers import *
from loss import *
from model import distance_parser


def get_args():
    parser = argparse.ArgumentParser(
        description='Syntactic distance based neural parser')
    parser.add_argument('--epc', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--bthsz', type=int, default=20)
    parser.add_argument('--hidsz', type=int, default=1200)
    parser.add_argument('--embedsz', type=int, default=400)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--dpout', type=float, default=0.3)
    parser.add_argument('--dpoute', type=float, default=0.1)
    parser.add_argument('--dpoutr', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--use_glove', action='store_true')
    parser.add_argument('--logfre', type=int, default=200)
    parser.add_argument('--devfre', type=int, default=-1)
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--datapath', type=str, default='../data/ptb')
    parser.add_argument('--savepath', type=str, default='../results')
    parser.add_argument('--filename', type=str, default=None)

    args = parser.parse_args()
    # set seed and return args
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.random.manual_seed(args.seed)
    return args


def evaluate(model, data, mode='valid'):
    import tempfile
    model.eval()
    if mode == 'valid':
        idxs, tags, stags, arcs, dsts = data.batchify(mode, 1)
        _, _, _, _, _, sents, trees = data.valid
    elif mode == 'test':
        idxs, tags, stags, arcs, dsts = data.batchify(mode, 1)
        _, _, _, _, _, sents, trees = data.test

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    print("Temp: {}, {}".format(temp_file_path, temp_targ_path))
    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    set_loss = 0.0
    set_counter = 0
    set_arc_prec = 0.0
    arc_counter = 0
    set_tag_prec = 0.0
    tag_counter = 0
    for _, (idx, tag, stag, arc, dst, sent, tree) in enumerate(
            zip(idxs, tags, stags, arcs, dsts, sents, trees)):

        if args.cuda:
            idx = idx.cuda()
            tag = tag.cuda()
            stag = stag.cuda()
            arc = arc.cuda()
            dst = dst.cuda()

        mask = (idx >= 0).float()
        idx = idx * mask.long()
        dstmask = (dst > 0).float()
        pred_dst, pred_arc, pred_tag = model(idx, stag, mask)

        loss = rankloss(pred_dst, dst, dstmask)
        set_loss += loss.item()
        set_counter += 1

        _, pred_arc_idx = torch.max(pred_arc, dim=-1)
        arc_prec = ((arc == pred_arc_idx).float() * dstmask).sum()
        set_arc_prec += arc_prec.item()
        arc_counter += dstmask.sum().item()

        _, pred_tag_idx = torch.max(pred_tag, dim=-1)
        pred_tag_idx[0], pred_tag_idx[-1] = -1, -1
        tag_prec = (tag == pred_tag_idx).float().sum()
        set_tag_prec += tag_prec.item()
        tag_counter += (tag != 0).float().sum().item()

        pred_tree = build_nltktree(
            pred_dst.data.squeeze().cpu().numpy().tolist()[1:-1],
            pred_arc_idx.data.squeeze().cpu().numpy().tolist()[1:-1],
            pred_tag_idx.data.squeeze().cpu().numpy().tolist()[1:-1],
            sent,
            ptb_parsed.arc_dictionary.idx2word,
            ptb_parsed.arc_dictionary.idx2word,
            ptb_parsed.stag_dictionary.idx2word,
            stags=stag.data.squeeze().cpu().numpy().tolist()[1:-1]
        )

        def process_str_tree(str_tree):
            return re.sub('[ |\n]+', ' ', str_tree)

        temp_tree_file.write(process_str_tree(str(pred_tree)) + '\n')
        temp_targ_file.write(process_str_tree(str(tree)) + '\n')

    # execute the evalb command:
    temp_tree_file.close()
    temp_targ_file.close()

    evalb_dir = os.path.join(os.getcwd(), "../EVALB")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    import subprocess
    subprocess.run(command, shell=True)
    fscore = FScore(math.nan, math.nan, math.nan)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
                break

    temp_path.cleanup()

    set_loss /= set_counter
    set_arc_prec /= arc_counter
    set_tag_prec /= tag_counter

    model.train()

    return (set_loss, set_arc_prec, set_tag_prec,
            fscore.precision, fscore.recall, fscore.fscore)


args = get_args()

if args.filename is None:
    filename = sorted(str(args)[10:-1].split(', '))
    filename = [i for i in filename if ('dir' not in i) and
                                       ('tblog' not in i) and
                                       ('fre' not in i) and
                                       ('cuda' not in i) and
                                       ('nlookback' not in i)]
    filename = __file__.split('.')[0].split('/')[-1] + '_' + \
               '_'.join(filename).replace('=', '') \
                                 .replace('/', '') \
                                 .replace('\'', '') \
                                 .replace('..', '') \
                                 .replace('\"', '')
else:
    filename = args.filename

if not os.path.isdir(args.savepath):
    os.mkdir(args.savepath)
parameter_filepath = os.path.join(args.savepath, filename + '.th')
print('model parth:', parameter_filepath)

print(args)
print("loading data ...")
ptb_parsed = PTBLoader(data_path=args.datapath, use_glove=args.use_glove)

wordembed = ptb_parsed.wordembed_matrix
args.vocab_size = len(ptb_parsed.dictionary)

train_log_template = 'epoch {:<3d} batch {:<4d} loss {:<.6f} rank {:<.6f} arc {:<.6f} tag {:<.6f}'
valid_log_template = \
    '*** epoch {:<3d}  \tloss     \tarc prec \ttag prec \tprecision\trecall   \tlf1      \n' \
    '{:10}DEV\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\n' \
    '{:10}TEST\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}'

if __name__ == '__main__':
    print("building model...")
    model = distance_parser(vocab_size=args.vocab_size,
                            embed_size=args.embedsz,
                            hid_size=args.hidsz,
                            arc_size=len(ptb_parsed.arc_dictionary),
                            stag_size=len(ptb_parsed.stag_dictionary),
                            window_size=args.window_size,
                            dropout=args.dpout,
                            dropoute=args.dpoute,
                            dropoutr=args.dpoutr,
                            wordembed=wordembed)
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, min_lr=0.000001)

    print(" ")
    numparams = sum([numpy.prod(i.size()) for i in model.parameters()])
    print("Number of params: {0}\n{1:35}{2:35}Size".format(
        numparams, 'Name', 'Shape'))  # this includes tied parameters
    print("---------------------------------------------------------------------------")
    for item in model.state_dict().keys():
        this_param = model.state_dict()[item]
        print("{:60}{!s:35}{}".format(
            item, this_param.size(), numpy.prod(this_param.size())))
    print(" ")

    # setting up training initial variables; checking out to previous model (if exist)
    best_valid_f1 = 0.0
    start_epoch = 0

    print("training ...")

    train_idxs, train_tags, train_stags, \
    train_arcs, train_distances, \
    train_sents, train_trees = ptb_parsed.batchify('train', args.bthsz)
    if args.devfre == -1:
        args.devfre = len(train_idxs)

    for epoch in range(start_epoch, start_epoch + args.epc):
        inds = list(range(len(train_idxs)))
        random.shuffle(inds)
        epc_train_idxs = [train_idxs[i] for i in inds]
        epc_train_tags = [train_tags[i] for i in inds]
        epc_train_stags = [train_stags[i] for i in inds]
        epc_train_arcs = [train_arcs[i] for i in inds]
        epc_train_distances = [train_distances[i] for i in inds]

        for ibatch, (idx, tag, stag, arc, dst) in \
                enumerate(
                    zip(
                        epc_train_idxs,
                        epc_train_tags,
                        epc_train_stags,
                        epc_train_arcs,
                        epc_train_distances,
                    )):

            if args.cuda:
                idx = idx.cuda()
                tag = tag.cuda()
                stag = stag.cuda()
                arc = arc.cuda()
                dst = dst.cuda()

            mask = (idx >= 0).float()
            idx = idx * mask.long()
            dstmask = (dst > 0).float()

            optimizer.zero_grad()
            pred_dst, pred_arc, pred_tag = model(idx, stag, mask)
            loss_rank = rankloss(pred_dst, dst, dstmask)
            loss_arc = arcloss(pred_arc, arc.view(-1))
            loss_tag = tagloss(pred_tag, tag.view(-1))

            loss = loss_rank + loss_arc + loss_tag
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            if (ibatch + 1) % args.logfre == 0:
                print(train_log_template.format(epoch, ibatch + 1, loss.item(),
                                                loss_rank.item(), loss_arc.item(),
                                                loss_tag.item()))

            #####

        print("Evaluating valid... ")
        valid_loss, valid_arc_prec, valid_tag_prec, \
        valid_precision, valid_recall, valid_f1 = evaluate(model, ptb_parsed, 'valid')
        print("Evaluating test... ")
        test_loss, test_arc_prec, test_tag_prec, \
        test_precision, test_recall, test_f1 = evaluate(model, ptb_parsed, 'test')
        print(valid_log_template.format(
            epoch,
            ' ', valid_loss, valid_arc_prec, valid_tag_prec,
            valid_precision, valid_recall, valid_f1,
            ' ', test_loss, test_arc_prec, test_tag_prec,
            test_precision, test_recall, test_f1))

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save({
                'epoch': epoch,
                'valid_loss': valid_loss,
                'valid_precision': valid_precision,
                'valid_recall': valid_recall,
                'valid_f1': valid_f1,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, parameter_filepath)

        scheduler.step(valid_f1)
