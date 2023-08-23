import argparse
import importlib
import numpy as np
import os
from tensorboardX import SummaryWriter
import time
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '..')

import aux_code.config as cfg
from aux_code.model_loaders import load_ft_model
from aux_code.models.large_i3d import freeze_bn
from aux_code.ucf101_dl import *


# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, data_loader, ft_model, criterion, criterion_temporal, optimizer, writer, use_cuda, lr, scaler, device_name, params):
    print(f'Train at epoch {epoch}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        writer.add_scalar('Learning Rate', lr, epoch)  
        print(f'Learning rate is: {param_group["lr"]}')
  
    losses = []
    predictions, gt = [], []

    # Set ft model to train.
    if params.arch == 'largei3d':
        freeze_bn(ft_model, 'ft_model')
    else:
        ft_model.train()

    for i, (inputs, label, _, _) in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.permute(0,2,1,3,4)

        if use_cuda:
            inputs = inputs.to(device=torch.device(device_name), non_blocking=True)
            label = torch.from_numpy(np.asarray(label)).to(device=torch.device(device_name), non_blocking=True)

        # Autocast automatic mixed precision.
        with autocast():
            if params.temporal_loss == 'trip':
                inputs1, inputs2, inputs3 = torch.split(inputs, [params.num_frames, params.num_frames, params.num_frames], dim=2)
            elif params.loss == 'con':
                inputs1, inputs2 = torch.split(inputs, [params.num_frames, params.num_frames], dim=2)
            else:
                inputs1 = inputs
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
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predictions.extend(torch.max(output, axis=1).indices.cpu().numpy())
            gt.extend(label.cpu().numpy())

        losses.append(loss.item())

        if i % 1000 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}', flush=True)
    
    print(f'Training Epoch: {epoch}, Loss: {np.mean(losses)}')
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    predictions = np.asarray(predictions)
    gt = np.asarray(gt)

    accuracy = ((predictions==gt).sum())/np.size(predictions)
    print(f'Training Accuracy at Epoch {epoch} is {accuracy*100:0.3f}%')

    del loss, inputs, output, label, feat1, inputs1
    if params.temporal_loss == 'trip':
        del feat2, feat3

    return ft_model, np.mean(losses), scaler


# Validation epoch.
def val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict, data_loader, ft_model, criterion, criterion_temporal, use_cuda, device_name, params):
    print(f'Validation at epoch {epoch}.')
    
    # Set model to eval.
    ft_model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, _) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        inputs = inputs.permute(0,2,1,3,4)
        
        if use_cuda:
            inputs = inputs.to(device=torch.device(device_name), non_blocking=True)
            label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).to(device=torch.device(device_name), non_blocking=True)

        with torch.no_grad():
            if params.temporal_loss == 'trip':
                inputs1, inputs2, inputs3 = torch.split(inputs, [params.num_frames, params.num_frames, params.num_frames], dim=2)
            elif params.loss == 'con':
                inputs1, inputs2 = torch.split(inputs, [params.num_frames, params.num_frames], dim=2)
            else:
                inputs1 = inputs
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

        predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())

        if i % 200 == 0:
            print(f'Validation Epoch {epoch}, Batch {i}, Loss : {np.mean(losses)}', flush=True)

    del loss, inputs, output, label, feat1, inputs1
    if params.temporal_loss == 'trip':
        del feat2, feat3

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:, 0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)

    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac} - Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)


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

    # Load in correct model file.
    if params.restart:
        saved_model_file = os.path.join(save_dir, 'model_temp.pth')
        
        if os.path.exists(saved_model_file):
            ft_model = load_ft_model(arch=params.arch, saved_model_file=saved_model_file, num_classes=params.num_classes)
            epoch1 = torch.load(saved_model_file)['epoch']
        else:
            print(f'No such model exists: {saved_model_file} :(')
            epoch1 = 1
    else:
        if not (params.saved_model == None or len(params.saved_model) == 0):
            print(f'Trying to load {params.saved_model}')
            ft_model = load_ft_model(arch=params.arch, saved_model_file=params.saved_model, num_classes=params.num_classes)
        else:
            print(f'It`s a baseline experiment!')
            # Load in fresh init ft_model.
            ft_model = load_ft_model(arch=params.arch, num_classes=params.num_classes, kin_pretrained=True)
        epoch1 = 1

    # Init loss function.
    if params.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'Loss function {params.loss} not yet implemented.')
    scaler = GradScaler()

    if params.temporal_loss == 'trip':
        criterion_temporal = nn.TripletMarginLoss()
    else:
        criterion_temporal = None

    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUS found!')
        ft_model = nn.DataParallel(ft_model, device_ids=devices)
        ft_model.cuda()
        criterion.cuda()
        if params.temporal_loss == 'trip':
            criterion_temporal.cuda()
    else:
        print('Only 1 GPU is available')
        ft_model.to(device=torch.device(device_name))
        criterion.to(device=torch.device(device_name))
        if params.temporal_loss == 'trip':
            criterion_temporal.to(device=torch.device(device_name))

    # Select optimizer.
    if params.opt_type == 'adam':
        optimizer = torch.optim.Adam(ft_model.parameters(), lr=params.learning_rate)
    elif params.opt_type == 'adamw':
        optimizer = torch.optim.AdamW(ft_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif params.opt_type == 'sgd':
        optimizer = torch.optim.SGD(ft_model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {params.opt_type} not yet implemented.')

    modes = list(range(params.num_modes))
    cropping_facs = params.cropping_facs

    val_array = params.val_array

    print(f'Base learning rate {params.learning_rate}')

    accuracy = 0
    best_score = 0
    best_acc = 0
    train_loss = 1000
    learning_rate = params.learning_rate
    orig_learning_rate = learning_rate
    scheduler_epoch = 0
    scheduler_step = 1

    for epoch in range(epoch1, params.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()

        if params.loss == 'con' or params.temporal_loss == 'trip':
            train_dataset = contrastive_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)
        else:
            train_dataset = single_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)

        if epoch == epoch1:
            print(f'Train dataset length: {len(train_dataset)}')
            print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')

        train_dataloader = DataLoader(
            train_dataset, 
            shuffle=True,
            batch_size=params.batch_size, 
            num_workers=params.num_workers,
            drop_last=True,
            collate_fn=collate_fn_train,
            pin_memory=True)

        # Warmup/LR scheduler.
        if params.lr_scheduler == 'cosine':
            learning_rate = params.cosine_lr_array[epoch-1]*orig_learning_rate
        elif params.warmup and epoch-1 < len(params.warmup_array):
            learning_rate = params.warmup_array[epoch-1]*orig_learning_rate
        elif params.lr_scheduler == 'loss_based':
            if 0.5 <= train_loss < 1.0:
                learning_rate = orig_learning_rate/2
            elif 0.1 <= train_loss < 0.5:
                learning_rate = orig_learning_rate/10
            elif train_loss < 0.1:
                learning_rate = orig_learning_rate/20
        elif params.lr_scheduler == 'patience_based':
            if scheduler_epoch == params.lr_patience:
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                print(f'Dropping learning rate to {learning_rate/(params.lr_reduce_factor**scheduler_step)} at epoch {epoch}.')
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                learning_rate = orig_learning_rate/(params.lr_reduce_factor**scheduler_step)
                scheduler_epoch = 0
                scheduler_step += 1

        ft_model, train_loss, scaler = train_epoch(epoch, train_dataloader, ft_model, criterion, criterion_temporal, optimizer, writer, use_cuda, learning_rate, scaler, device_name, params)

        if train_loss < best_score:
            best_score = train_loss
            scheduler_epoch = 0
        else:
            scheduler_epoch += 1


        # Validation epoch.
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []

            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    if params.loss == 'con' or params.temporal_loss == 'trip':
                        validation_dataset = contrastive_val_dataloader(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    else:
                        validation_dataset = single_val_dataloader(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, drop_last=True, collate_fn=collate_fn_val)
                    if val_iter == 0:
                        print(f'Validation dataset length: {len(validation_dataset)}')
                        print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')
                    pred_dict, label_dict, accuracy, loss = val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict, validation_dataloader, ft_model, criterion, criterion_temporal, use_cuda, device_name, params)
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
                    print(f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]}, is {accuracy_all*100:.3f}%')

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
            print(f'Overall Accuracy is for epoch {epoch} is {accuracy*100:.3f}%')

            if accuracy > best_acc:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(accuracy)[:6]}.pth')
                states = {
                    'epoch': epoch + 1,
                    'amp_scaler': scaler,
                    'ft_model_state_dict': ft_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)
                best_acc = accuracy

        # Temp saving.
        save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'amp_scaler' : scaler,
            'ft_model_state_dict': ft_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        if params.lr_scheduler != 'cosine' and learning_rate < 1e-12 and epoch > 10:
            print(f'Learning rate is very low now, stopping the training.')
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--params", dest='params', type=str, required=False, default='params_action.py', help='params')
    parser.add_argument("--devices", dest='devices', action='append', type=int, required=False, default=None, help='devices should be a list')

    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    train_classifier(params, args.devices)
