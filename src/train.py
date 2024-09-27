# -*- coding: utf-8 -*-
import os
import gc
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from optimisers import get_optimiser

from toolsdev import aucPerformance

def pretrain(encoder, dataloaders, args):
    ''' Pretrain script - SCLM

        Pretrain the autoencoder with a Contrastive InfoNCE and RE Loss.
    '''
    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder,), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = args.base_lr

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.00001, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.n_epochs)

    ''' Loss / Criterion '''
    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion = nn.MSELoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf


    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()

        sample_count = 0
        run_loss = 0

        # Print setup
        if args.print_progress:
            logging.info('\nEpoch {}/{}:'.format(epoch+1, args.n_epochs))
            train_dataloader = tqdm(dataloaders['pretrain'])
        else:
            train_dataloader = dataloaders['pretrain']

        ''' epoch loop '''
        for i, (inputs, _) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            optimiser.zero_grad()

            # retrieve the two pieces of data
            x_i, x_j = torch.split(inputs, [1, 1], dim=1)
            x_i = torch.squeeze(x_i, dim=1)
            x_j = torch.squeeze(x_j, dim=1)

            # Get the encoder representation
            logit, label, mse = encoder(x_i, x_j)

            logit_label = label
            label = label.unsqueeze(1).float()
            loss = args.c_lr * criterion1(logit, logit_label) + (1 - args.c_lr) * criterion(mse, label)
            # loss = criterion(mse, label)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (args.learning_rate - args.base_lr) * \
                (float(epoch+1) / args.warmup_epochs) + args.base_lr
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            # args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            best_valid_loss = epoch_pretrain_loss

            # saving using process (rank) 0 only as all processes are in sync
            state = {
                #'args': args,
                'sclm': encoder.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, args.checkpoint_dir)

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def score_finetune(encoder, dataloaders, args, rauc, ap, t):
    ''' Finetune Train script - SEAD

        Finetune Training encoder and train the SEAD with APM.
    '''

    mode = 'finetune'

    ''' Optimisers '''
    # optimise
    optimiser = get_optimiser((encoder,), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = args.base_lr_finetune

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0001, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.n_epochs)

    ''' Loss / Criterion '''
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # criterion = nn.MSELoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    max_model_save_sum = float('-inf')

    ''' Pretrain loop '''
    for epoch in range(args.finetune_epochs):

        # Freeze the encoder, train classification
        encoder.train()

        sample_count = 0
        run_loss = 0
        run_top1 = 0.0
        run_top5 = 0.0

        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.finetune_epochs))
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        ''' epoch loop '''
        for i, (inputs, target) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            inputs = torch.squeeze(inputs, dim=1)
            output, score, sub_result, mse = encoder(inputs)
            score = score.squeeze()


            def multi_loss(y_true, y_pred, sub_result):
                confidence_margin = 1.5
                dev = y_pred
                inlier_loss = torch.abs(dev)
                outlier_loss = torch.abs(torch.max(confidence_margin - dev, torch.zeros_like(dev)))
                sub_nor = torch.norm(sub_result, p=2, dim=1)
                outlier_sub_loss = torch.abs(torch.max(confidence_margin - sub_nor, torch.zeros_like(sub_nor)))
                loss1 = (1 - y_true) * (inlier_loss + sub_nor) + y_true * (outlier_loss + outlier_sub_loss)
                loss1 = loss1.mean()

                return loss1

            loss = multi_loss(target, score, sub_result)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

            predicted = output.argmax(1)


            acc = (predicted == target).sum().item() / target.size(0)

            run_top1 += acc

            _, output_topk = output.topk(1, 1, True, True)

            acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                        ).sum().item() / target.size(0)  # num corrects

            run_top5 += acc_top5

        epoch_finetune_loss = run_loss / len(dataloaders['train'])  # sample_count

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        # If args.warmup_epochs > 0 and the current epoch + 1 is less than or equal to args.warmup_epochs, the learning rate is pre-warmed. During the pre-warming phase, the learning rate is gradually increased from args.base_lr_finetune to args.learning_rate_finetune
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (args.learning_rate_finetune - args.base_lr_finetune) * \
                (float(epoch+1) / args.warmup_epochs) + args.base_lr_finetune
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_finetune_loss))
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        encoder.eval()

        model_save_index, max_model_save_sum = evaluate(
            encoder, dataloaders, 'test', epoch, args, rauc, ap, t, max_model_save_sum)

        if model_save_index is True:
            print('best model save....')
            state = {
                # 'args': args,
                'base_encoder': encoder.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch,
            }

            torch.save(state, (args.checkpoint_dir[:-3] + "_finetune.pt"))

        epoch_finetune_loss = None

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def evaluate(encoder, dataloaders, mode, finetune_epochs, args, rauc, ap, t, max_model_save_sum):
    ''' Evaluate script - SEAD

        Evaluate the encoder.
    '''

    model_save_index = False

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder
    encoder.eval()

    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0

    if args.print_progress:
        eval_dataloader = tqdm(dataloaders[mode])
    else:
        eval_dataloader = dataloaders[mode]

    all_predicted = []
    all_target = []

    ''' epoch loop '''
    for i, (inputs, target) in enumerate(eval_dataloader):

        # Do not compute gradient for encoder
        encoder.zero_grad()

        inputs = inputs.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        # Forward pass
        inputs = torch.squeeze(inputs, dim=1)
        # output = encoder(inputs)

        output, score, sub_result, mse = encoder(inputs)
        score = score.squeeze()

        # loss = criterion(score, target.float())
        loss = criterion(output.float(), target.long())

        torch.cuda.synchronize()

        sample_count += inputs.size(0)

        run_loss += loss.item()

        predicted = output.argmax(1)

        # all_predicted.append(predicted)
        all_predicted.append(score)
        all_target.append(target)

        acc = (predicted == target).sum().item() / target.size(0)

        run_top1 += acc

        _, output_topk = output.topk(1, 1, True, True)

        acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / target.size(0)  # num corrects

        run_top5 += acc_top5

    best_valid_loss = np.inf

    if mode == 'test':
        all_predicted = torch.cat(all_predicted, dim=0)
        all_target = torch.cat(all_target, dim=0)
        rauc[t][finetune_epochs], ap[t][finetune_epochs] = aucPerformance(all_predicted, all_target)
        logging.info(
            'AUC-ROC: {:.4f} \t PR: {:.4f} '.format(rauc[t][finetune_epochs], ap[t][finetune_epochs]))

        model_save_sum = rauc[t][finetune_epochs] + ap[t][finetune_epochs]
        if model_save_sum > max_model_save_sum:
            max_model_save_sum = model_save_sum
            model_save_index = True

        # If the current round is the final round of fine-tuning, calculate the maximum AUC and PR for each model in all fine-tuning rounds
        max_rauc_ap_sum = float('-inf')
        if finetune_epochs + 1 == args.finetune_epochs:
            # Select the best result in the fine-tuning stage
            for j in range(args.finetune_epochs):
                current_sum = rauc[t][j] + ap[t][j]
                if current_sum > max_rauc_ap_sum:
                    max_rauc_ap_sum = current_sum
                    max_rauc = rauc[t][j]
                    max_ap = ap[t][j]
            rauc[t][finetune_epochs + 1] = max_rauc
            ap[t][finetune_epochs + 1] = max_ap

            for m in range(t + 1):
                if rauc[m][args.finetune_epochs] != 0:
                    logging.info(
                        ' epoch[{:d}] \t per max AUC-ROC: {:.4f} \t per max PR: {:.4f} '.format(m, rauc[m][finetune_epochs + 1], ap[m][finetune_epochs + 1]))

            if t+1 == args.runs:
                max_rauc_ap_sum_all = float('-inf')
                for n in range(args.runs):
                    current_sum = rauc[n][args.finetune_epochs] + ap[n][args.finetune_epochs]
                    if current_sum > max_rauc_ap_sum_all:
                        max_rauc_ap_sum_all = current_sum
                        max_rauc_all = rauc[n][args.finetune_epochs]
                        max_ap_all = ap[n][args.finetune_epochs]
                logging.info(
                    'Final: \t best AUC-ROC: {:.4f} \t best PR: {:.4f} '.format(max_rauc_all, max_ap_all))
                # rauc[t][finetune_epochs + 1] = max_rauc_all
                # ap[t][finetune_epochs + 1] = max_ap_all



    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

    return model_save_index, max_model_save_sum
