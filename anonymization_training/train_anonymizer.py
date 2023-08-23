import argparse
import importlib
import itertools
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import time
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import sys
sys.path.insert(0, '..')

from aux_code import config as cfg
from aux_code.model_loaders import load_ft_model, load_fa_model, load_fb_model
from aux_code.ucf101_dl import *
from aux_code.nt_xent_original import NTXentLoss
from aux_code.vispr_dl import  vispr_dataset, vispr_ssl_dataset


# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, dataloader_vispr, dataloader_video, ft_model, fa_model, fb_model, criterion_ft, criterion_temporal_ft, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda, learning_rate_fa, learning_rate_fb, learning_rate_ft, device_name, params):
    print(f'Train at epoch {epoch}')
    for param_group_fa in optimizer_fa.param_groups:
        param_group_fa['lr'] = learning_rate_fa
    for param_group_fb in optimizer_fb.param_groups:
        param_group_fb['lr'] = learning_rate_fb
    for param_group_ft in optimizer_ft.param_groups:
        param_group_ft['lr'] = learning_rate_ft
        
    writer.add_scalar('Learning Rate Fa', learning_rate_fa, epoch)
    writer.add_scalar('Learning Rate Fb', learning_rate_fb, epoch)
    writer.add_scalar('Learning Rate Ft', learning_rate_ft, epoch)  
    print(f'Learning rate of fa is: {param_group_fa["lr"]}')
    print(f'Learning rate of fb is: {param_group_fb["lr"]}')
    print(f'Learning rate of ft is: {param_group_ft["lr"]}')
    
    losses_fa, losses_fb, losses_ft, losses_temporal = [], [], [], []
    # predictions, gt = [], []

    step = 1
    for i, (data1, data2) in enumerate(zip(dataloader_vispr, dataloader_video)):
        inputs_video, labels_video = data2[0:2]
        if inputs_video == None:
            continue
        inputs_vispr = [data1[ii] for ii in range(2)]
        inputs_video = inputs_video.permute(0,2,1,3,4)

        if use_cuda:
            inputs_vispr = [inputs_vispr[ii].to(device=torch.device(device_name), non_blocking=True) for ii in range(2)]
            # Do NOT use VISPR privacy labels in training.
            # labels_vispr = torch.from_numpy(np.asarray(data1[2])).float().cuda()
            inputs_video = inputs_video.to(device=torch.device(device_name), non_blocking=True)
            labels_video = torch.from_numpy(np.asarray(labels_video)).type(torch.LongTensor).to(device=torch.device(device_name), non_blocking=True)

        optimizer_fa.zero_grad()
        optimizer_fb.zero_grad()
        optimizer_ft.zero_grad()

        # Step-1: Update fa.
        if step == 1:
            # Train fa model, not ft and fb.
            fa_model.train()
            ft_model.eval()
            fb_model.eval()

            # Autocast automatic mixed precision.
            with autocast():
                # Get anonymous reconstruction from fa, input to fb.
                output1 = [fb_model(fa_model(inputs_vispr[ii])) for ii in range(2)]
                # Contrastive loss function for SSL.
                con_loss_criterion = NTXentLoss(device='cuda', batch_size=output1[0].shape[0], temperature=0.1, use_cosine_similarity=False)
                # Compute losses.
                loss_fb = con_loss_criterion(output1[0], output1[1])

                # Split original shape up.
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_video.shape
                # Reshape video input.
                inputs_video = inputs_video.reshape(-1, inputs_video.shape[1], inputs_video.shape[3], inputs_video.shape[4])

                # Get anonymous reconstruction from fa, input to ft.
                anon_input = fa_model(inputs_video).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
                if params.temporal_loss == 'trip':
                    inputs1, inputs2, inputs3 = torch.split(anon_input, [params.num_frames, params.num_frames, params.num_frames], dim=2)
                elif params.loss == 'con':
                    inputs1, inputs2 = torch.split(anon_input, [params.num_frames, params.num_frames], dim=2)
                else:
                    inputs1 = anon_input
                output, feat1 = ft_model(inputs1)
                if params.loss == 'con':
                    _, feat2 = ft_model(inputs2)
                    feature = torch.stack([F.normalize(feat1, dim=1), F.normalize(feat2, dim=1)], dim=1)
                    # Compute loss.
                    loss_ft = criterion_ft(feature, labels_video)
                else:
                    # Compute loss.
                    loss_ft = criterion_ft(output, labels_video)

                if params.temporal_loss == 'trip':
                    if params.loss != 'con':
                        _, feat2 = ft_model(inputs2)
                    _, feat3 = ft_model(inputs3)

                    # Compute temporal loss, append to loss.
                    loss_temporal = criterion_temporal_ft(feat1, feat2, feat3)
                    loss_ft = loss_ft + params.temporal_loss_weight*loss_temporal

                # Combine losses into single fa loss.
                loss_fa = -params.fb_loss_weight*loss_fb + params.ft_loss_weight*loss_ft
            
            losses_fa.append(loss_fa.item())
            loss_fa.backward()
            optimizer_fa.step()

            # Set to step 2 to update other networks.
            step = 2

            if i % 100 == 0:
                print(f'Training Epoch {epoch}, Batch {i}, loss_fa: {np.mean(losses_fa) :.5f}, loss_temporal: {np.mean(losses_temporal) :.5f}', flush = True)

            # Skip step 2 for this batch, go to next batch.
            continue
        
        # Step-2: Update ft and fb.
        if step == 2:
            # Set ft and fb to train, fa to eval.
            fa_model.eval()
            fb_model.train()
            ft_model.train()

            # Run inputs through fa_model.
            with torch.no_grad():
                # Split original shape up.
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_video.shape
                # Reshape video input.
                inputs_video = inputs_video.reshape(-1, inputs_video.shape[1], inputs_video.shape[3], inputs_video.shape[4])
                input1 = [fa_model(inputs_vispr[ii]) for ii in range(2)]
                input2 = fa_model(inputs_video).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)

            # Autocast automatic mixed precision.
            with autocast():
                # Get anonymous reconstruction from fa, input to fb.
                output1 = [fb_model(x) for x in input1]
                # Contrastive loss function for SSL.
                con_loss_criterion = NTXentLoss(device='cuda', batch_size=output1[0].shape[0], temperature=0.1, use_cosine_similarity=False)
                # Compute losses.
                loss_fb = con_loss_criterion(output1[0], output1[1])

                # Get anonymous reconstruction from fa, input to ft.
                if params.temporal_loss == 'trip':
                    inputs1, inputs2, inputs3 = torch.split(input2, [params.num_frames, params.num_frames, params.num_frames], dim=2)
                elif params.loss == 'con':
                    inputs1, inputs2 = torch.split(input2, [params.num_frames, params.num_frames], dim=2)
                else:
                    inputs1 = input2
                output2, feat1 = ft_model(inputs1)
                if params.loss == 'con':
                    _, feat2 = ft_model(inputs2)
                    feature = torch.stack([F.normalize(feat1, dim=1), F.normalize(feat2, dim=1)], dim=1)
                    # Compute loss.
                    loss_ft = criterion_ft(feature, labels_video)
                else:
                    # Compute loss.
                    loss_ft = criterion_ft(output2, labels_video)

                if params.temporal_loss == 'trip':
                    if params.loss != 'con':
                        _, feat2 = ft_model(inputs2)
                    _, feat3 = ft_model(inputs3)

                    # Compute temporal loss, append to loss.
                    loss_temporal = criterion_temporal_ft(feat1, feat2, feat3)
                    loss_ft = loss_ft + params.temporal_loss_weight*loss_temporal

            losses_ft.append(float(loss_ft.item()))
            losses_fb.append(float(loss_fb.item()))
            if params.temporal_loss == 'trip':
                losses_temporal.append(float(loss_temporal.item()))

            loss_fb.backward()
            loss_ft.backward()
            optimizer_fb.step()
            optimizer_ft.step()
            # Set to step 2 to update other network.
            step = 1
            if i % 100 == 0:
                print(f'Training Epoch {epoch}, Batch {i}, loss_fb: {np.mean(losses_fb) :.5f}, loss_ft: {np.mean(losses_ft) :.5f}, loss_temporal: {np.mean(losses_temporal) :.5f}', flush = True)
            continue
    
    print(f'Training Epoch: {epoch}, loss_fa: {np.mean(losses_fa):.4f}, loss_fb: {np.mean(losses_fb):.4f}, loss_ft: {np.mean(losses_ft):.4f}')

    writer.add_scalar('Training loss_fa', np.mean(losses_fa), epoch)
    writer.add_scalar('Training loss_fb', np.mean(losses_fb), epoch)
    writer.add_scalar('Training loss_ft', np.mean(losses_ft), epoch)
    if params.temporal_loss == 'trip':
        writer.add_scalar('Training temporal_loss', np.mean(losses_temporal), epoch)

    del loss_fb, loss_ft, loss_fa, inputs_vispr, inputs_video, inputs1, output1, output2, anon_input, input1, input2
    if params.temporal_loss == 'trip':
        del inputs2, inputs3

    return fa_model, fb_model, ft_model


# Validation epoch.
def val_epoch_video(epoch, mode, cropping_fac, pred_dict, label_dict, data_loader, ft_model, fa_model, criterion, criterion_temporal, use_cuda, device_name, params):
    print(f'Validation at epoch {epoch}.')
    
    # Set models to eval.
    ft_model.eval()
    fa_model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, _) in enumerate(data_loader):
        if vid_path == None:
            continue
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        inputs = inputs.permute(0,2,1,3,4)
        
        if use_cuda:
            inputs = inputs.to(device=torch.device(device_name), non_blocking=True)
            label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).to(device=torch.device(device_name), non_blocking=True)

        with torch.no_grad():
            # Split original shape up.
            ori_bs, ori_t, ori_c, ori_h, ori_w = inputs.shape
            # Reshape video input.
            inputs = inputs.reshape(-1, inputs.shape[1], inputs.shape[3], inputs.shape[4])
            anon_input = fa_model(inputs).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)

            if params.temporal_loss == 'trip':
                inputs1, inputs2, inputs3 = torch.split(anon_input, [params.num_frames, params.num_frames, params.num_frames], dim=2)
            elif params.loss == 'con':
                inputs1, inputs2 = torch.split(anon_input, [params.num_frames, params.num_frames], dim=2)
            else:
                inputs1 = anon_input
            output, feat1 = ft_model(inputs1)
            if params.loss == 'con':
                _, feat2 = ft_model(inputs2)
                feature = torch.stack([F.normalize(feat1, dim=1), F.normalize(feat2, dim=1)], dim=1)
                # Compute loss.
                loss = criterion(feature, label)
            else:
                # Compute loss.
                loss = criterion(output, label)

            if params.temporal_loss == 'trip':
                if params.loss != 'con':
                    _, feat2 = ft_model(inputs2)
                _, feat3 = ft_model(inputs3)

                # Compute temporal loss, append to loss.
                loss_temporal = criterion_temporal(feat1, feat2, feat3)
                loss = loss + params.temporal_loss_weight*loss_temporal

        losses.append(loss.item())

        predictions.extend(nn.functional.softmax(output, dim=1).cpu().data.numpy())

        if i % 200 == 0:
            print(f'Validation Epoch {epoch}, Batch {i}, Loss : {np.mean(losses)}', flush=True)
        
    del inputs, output, label, loss, anon_input
    if params.temporal_loss == 'trip':
        del inputs2, inputs3

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1) 
    c_pred = pred_array[:, 0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])
        else:
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)

    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac} - Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)


# Visualize anonymized reconstruction on validation epoch.
def val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model):
    with torch.inference_mode():
        for inputs, _, _ in validation_dataloader:
            if len(inputs.shape) == 1:
                continue
            inputs = inputs.cuda()
            image_full_name = os.path.join(save_dir, f'combined_epoch{epoch}.png')
            outputs = fa_model(inputs)
            vis_image = torch.cat([inputs, outputs], dim=0)
            save_image(vis_image, image_full_name, padding=5, nrow=int(inputs.shape[0]))
            return


# Main code. 
def train_classifier(params, devices):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')
    # Empty cuda cache.
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    writer = SummaryWriter(os.path.join(cfg.logs, str(params.run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load pretrained reconstruction fa_model.
    fa_model = load_fa_model(arch=params.arch_fa, saved_model_file=params.saved_model_fa)
    # Load in pretrained ft_model. 
    ft_model = load_ft_model(arch=params.arch_ft, saved_model_file=params.saved_model_ft, num_classes=params.num_classes, kin_pretrained=True if params.saved_model_ft is None else False)
    # Load in pretrained fb_model.
    fb_model = load_fb_model(arch=params.arch_fb, saved_model_file=params.saved_model_fb, ssl=True)

    epoch1 = 1

    learning_rate_fa = params.learning_rate_fa
    learning_rate_fb = params.learning_rate_fb
    learning_rate_ft = params.learning_rate_ft

    # Init loss functions.
    # TODO: con loss?
    if params.loss == 'con':
        criterion_ft = None
    elif params.loss == 'ce':
        criterion_ft = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'Loss function {params.loss} not yet implemented.')

    if params.temporal_loss == 'trip':
        criterion_temporal_ft = nn.TripletMarginLoss(margin=params.triplet_loss_margin)
    else:
        criterion_temporal_ft = None

    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUS found!')
        ft_model = nn.DataParallel(ft_model, device_ids=devices)
        fa_model = nn.DataParallel(fa_model, device_ids=devices)
        fb_model = nn.DataParallel(fb_model, device_ids=devices)
        ft_model.cuda()
        fa_model.cuda()
        fb_model.cuda()
        criterion_ft.cuda()
        if params.temporal_loss == 'trip':
            criterion_temporal_ft.cuda()
    else:
        print('Only 1 GPU is available')
        ft_model.to(device=torch.device(device_name))
        fa_model.to(device=torch.device(device_name))
        fb_model.to(device=torch.device(device_name))
        criterion_ft.to(device=torch.device(device_name))
        if params.temporal_loss == 'trip':
            criterion_temporal_ft.to(device=torch.device(device_name))

    # Select optimizer.
    if params.opt_type == 'adam':
        optimizer_fa = torch.optim.Adam(fa_model.parameters(), lr=params.learning_rate_fa)
        optimizer_fb = torch.optim.Adam(fb_model.parameters(), lr=params.learning_rate_fb)
        optimizer_ft = torch.optim.Adam(ft_model.parameters(), lr=params.learning_rate_ft)
    elif params.opt_type == 'adamw':
        optimizer_fa = torch.optim.AdamW(fa_model.parameters(), lr=params.learning_rate_fa, weight_decay=params.weight_decay)
        optimizer_fb = torch.optim.AdamW(fb_model.parameters(), lr=params.learning_rate_fb, weight_decay=params.weight_decay)
        optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=params.learning_rate_ft, weight_decay=params.weight_decay)
    elif params.opt_type == 'sgd':
        optimizer_fa = torch.optim.SGD(fa_model.parameters(), lr=params.learning_rate_fa, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer_fb = torch.optim.SGD(fb_model.parameters(), lr=params.learning_rate_fb, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer_ft = torch.optim.SGD(ft_model.parameters(), lr=params.learning_rate_ft, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {params.opt_type} not yet implemented.')

    train_dataset_vispr = vispr_ssl_dataset(data_split='train', shuffle=True, data_percentage=params.data_percentage_vispr)
    train_dataloader_vispr = DataLoader(train_dataset_vispr, batch_size=params.batch_size_vispr, shuffle=True, num_workers=params.num_workers)
    print(f'VISPR Train dataset length: {len(train_dataset_vispr)}')
    print(f'VISPR Train dataset steps per epoch: {len(train_dataset_vispr)/params.batch_size_vispr}')

    modes = list(range(params.num_modes))
    cropping_facs = params.cropping_facs
    modes, cropping_fac = list(zip(*itertools.product(modes, cropping_facs)))

    val_array = [1, 5, 10, 12, 15, 20, 25, 30, 35] + [40 + x*2 for x in range(30)]

    print(f'Num modes {len(modes)}')
    print(f'Cropping fac {cropping_facs}')
    print(f'Base learning rate {params.learning_rate}')

    accuracy = 0


    for epoch in range(epoch1, params.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()

        if params.loss == 'con' or params.temporal_loss == 'trip':
            train_dataset_video = contrastive_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)
        else:
            train_dataset_video = single_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)

        if epoch == epoch1:
            print(f'Video Train dataset length: {len(train_dataset_video)}')
            print(f'Video Train dataset steps per epoch: {len(train_dataset_video)/params.batch_size}')

        train_dataloader_video = DataLoader(
            train_dataset_video, 
            shuffle=True,
            batch_size=params.batch_size, 
            num_workers=params.num_workers,
            collate_fn=collate_fn_train,
            pin_memory=True)

        # Warmup/LR scheduler.
        # if params.lr_scheduler == 'cosine':
        #     learning_rate = params.cosine_lr_array[epoch-1]*orig_learning_rate
        # elif params.warmup and epoch-1 < len(params.warmup_array):
        #     learning_rate = params.warmup_array[epoch-1]*orig_learning_rate
        # elif params.lr_scheduler == 'loss_based':
        #     if 0.5 <= train_loss < 1.0:
        #         learning_rate = orig_learning_rate/2
        #     elif 0.1 <= train_loss < 0.5:
        #         learning_rate = orig_learning_rate/10
        #     elif train_loss < 0.1:
        #         learning_rate = orig_learning_rate/20
        # elif params.lr_scheduler == 'patience_based':
        #     if scheduler_epoch == params.lr_patience:
        #         print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
        #         print(f'Dropping learning rate to {learning_rate/(params.lr_reduce_factor**scheduler_step)} at epoch {epoch}.')
        #         print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
        #         learning_rate = orig_learning_rate/(params.lr_reduce_factor**scheduler_step)
        #         scheduler_epoch = 0
        #         scheduler_step += 1

        fa_model, fb_model, ft_model = train_epoch(epoch, train_dataloader_vispr, train_dataloader_video, ft_model, fa_model, fb_model, criterion_ft, criterion_temporal_ft, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda, learning_rate_fa, learning_rate_fb, learning_rate_ft, device_name, params)

        validation_dataset_vispr = vispr_dataset(data_split='test', shuffle=True, data_percentage=0.05)
        validation_dataloader = DataLoader(validation_dataset_vispr, batch_size=16, shuffle=True, num_workers=params.num_workers)
        val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model)
        # Validation epoch.
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []

            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    if params.loss == 'con' or params.temporal_loss == 'trip':
                        validation_dataset_video = contrastive_val_dataloader(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    else:
                        validation_dataset_video = single_val_dataloader(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    validation_dataloader_video = DataLoader(validation_dataset_video, batch_size=params.v_batch_size, num_workers=params.num_workers, collate_fn=collate_fn_val)
                    if val_iter == 0:
                        print(f'Video Validation dataset length: {len(validation_dataset_video)}')
                        print(f'Video Validation dataset steps per epoch: {len(validation_dataset_video)/params.v_batch_size}')
                    pred_dict, label_dict, accuracy, loss = val_epoch_video(epoch, mode, cropping_fac, pred_dict, label_dict, validation_dataloader_video, ft_model, fa_model, criterion_ft, criterion_temporal_ft, use_cuda, device_name, params)
                    val_losses.append(loss)

                    predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
                    ground_truth = []
                    for entry, key in enumerate(pred_dict.keys()):
                        predictions[entry] = np.mean(pred_dict[key], axis=0)

                    for key in label_dict.keys():
                        ground_truth.append(label_dict[key])

                    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
                    c_pred = pred_array[:, 0]

                    correct_count = np.sum(c_pred==ground_truth)
                    accuracy_all = float(correct_count)/len(c_pred)
                    print(f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]}, cropping_fac {cropping_fac}is {accuracy_all*100:.3f}%')

            val_loss = np.mean(val_losses)
            predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
            ground_truth = []

            for entry, key in enumerate(pred_dict.keys()):
                predictions[entry] = np.mean(pred_dict[key], axis=0)

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
            c_pred = pred_array[:,0]

            correct_count = np.sum(c_pred==ground_truth)
            accuracy = float(correct_count)/len(c_pred)
            print(f'Val loss for epoch {epoch} is {val_loss}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', accuracy, epoch)
            print(f'Overall Ft accuracy for epoch {epoch} is {accuracy*100:.3f}%')

            if accuracy > 0.6:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} has above 60% acc for {params.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(accuracy)[:6]}.pth')
                states = {
                    'epoch': epoch + 1,
                    'fa_model_state_dict': fa_model.state_dict(),
                    'fb_model_state_dict': fb_model.state_dict(),
                    'ft_model_state_dict': ft_model.state_dict()
                }
                torch.save(states, save_file_path)

        # We will save optimizer weights for each temp model, not all saved models to reduce the storage.
        save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'fa_model_state_dict': fa_model.state_dict(),
            'fb_model_state_dict': fb_model.state_dict(),
            'ft_model_state_dict': ft_model.state_dict()
        }
        torch.save(states, save_file_path)

        # Save every 3 to save space.
        if epoch % 3 == 0:
            save_file_path = os.path.join(save_dir, f'model_{epoch}.pth')
            states = {
                'epoch': epoch + 1,
                'fa_model_state_dict': fa_model.state_dict(),
                'fb_model_state_dict': fb_model.state_dict(),
                'ft_model_state_dict': ft_model.state_dict(),
                'optimizer_fa': optimizer_fa.state_dict(),
                'optimizer_fb': optimizer_fb.state_dict(),
                'optimizer_ft': optimizer_ft.state_dict()
            }
            torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--params", dest='params', type=str, required=False, default='params_anonymization.py', help='params')
    parser.add_argument("--devices", dest='devices', action='append', type=int, required=False, default=None, help='devices should be a list')

    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', ''))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    train_classifier(params, args.devices)
