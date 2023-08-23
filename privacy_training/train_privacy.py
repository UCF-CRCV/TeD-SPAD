import argparse
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import traceback

import params_privacy as params

import sys
sys.path.insert(0, '..')

from aux_code import config as cfg
from aux_code.model_loaders import load_fa_model, load_fb_model
from aux_code.vispr_dl import vispr_dataset


# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, data_loader, fa_model, fb_model, anon, criterion, optimizer, writer, use_cuda, learning_rate):
    print(f'Train Epoch {epoch}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        writer.add_scalar('Learning Rate', learning_rate, epoch)  
        print(f'Learning rate is: {param_group["lr"]}')
  
    losses= []

    fa_model.eval()
    fb_model.train()

    for i, (inputs, label, _) in enumerate(data_loader):
        optimizer.zero_grad()

        if use_cuda:
            inputs = inputs.cuda()
            label = torch.from_numpy(np.asarray(label)).float().cuda()
        
        if anon:
            output = fb_model(fa_model(inputs))
        else:
            output = fb_model(inputs)
        loss = criterion(output, label)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if i % 100 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
        
    print(f'Training Epoch: {epoch}, Loss: {np.mean(losses):.4f}')
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, inputs, output, label

    return fb_model, np.mean(losses)


# Validation epoch.
def val_epoch(epoch, validation_dataloader, fa_model, fb_model, anon, criterion, use_cuda, writer):
    fa_model.eval()
    fb_model.eval()
    losses = []
    predictions, ground_truth = [], []
    vid_paths = []
    label_dict, pred_dict = {}, {}

    for i, (inputs, label, vid_path) in enumerate(validation_dataloader):
        if len(inputs.shape) != 1:
            if use_cuda:
                inputs = inputs.cuda()
                label = torch.from_numpy(np.asarray(label)).float().cuda()

            with torch.no_grad():
                if anon:
                    output = fb_model(fa_model(inputs))
                else:
                    output = fb_model(inputs)
                loss = criterion(output, label)
                losses.append(loss.item())

            predictions.extend(output.cpu().data.numpy())
            vid_paths.extend(vid_path)
            ground_truth.extend(label.cpu().data.numpy())

            if i % 100 == 0:
                print(f'Validation Epoch {epoch}, Batch {i} - Loss : {np.mean(losses)}', flush=True)
        
    del inputs, output, label, loss 
    print(f'Validation Epoch {epoch} - Loss : {np.mean(losses)}')
    writer.add_scalar('Validation Loss', np.mean(losses), epoch)

    ground_truth = np.asarray(ground_truth)
    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))
    predictions = np.asarray(predictions)
    try:
        print(f'GT shape before putting in ap: {ground_truth.shape}')
        print(f'pred shape before putting in ap: {predictions.shape}')
    except:
        print(f'GT len before putting in ap: {len(ground_truth)}')
        print(f'pred len before putting in ap: {len(predictions)}')

    ap = average_precision_score(ground_truth, predictions, average=None)
    
    print(f'Macro f1 is {np.mean(f1)}')
    print(f'Macro prec is {np.mean(prec)}')
    print(f'Macro recall is {np.mean(recall)}')
    print(f'Classwise AP is {ap}')
    print(f'Macro AP is {np.mean(ap)}')

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])
        else:
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    return pred_dict, label_dict, np.mean(ap)
    

# Main code loop.
def train_classifier(run_id, saved_model, arch):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')
    use_cuda = True
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fa_model = load_fa_model(arch='unet++', saved_model_file=saved_model) #UNet(n_channels = 3, n_classes=3)
    # Freeze model weights.
    for param in fa_model.parameters():
        param.requires_grad = False
    
    # Initialize privacy prediction model.
    fb_model = load_fb_model(arch=arch, saved_model_file=None, ssl=False, pretrained=False, num_pa=params.num_pa)
    epoch1 = 1

    criterion = nn.BCEWithLogitsLoss()

    if torch.cuda.device_count() > 1:
        print(f'Multiple GPUS found!')
        fa_model = nn.DataParallel(fa_model)
        fb_model = nn.DataParallel(fb_model)
        criterion.cuda()
        fb_model.cuda()
        fa_model.cuda()
    else:
        print('Only 1 GPU is available')
        criterion.cuda()
        fb_model.cuda()
        fa_model.cuda()

    optimizer = optim.Adam(fb_model.parameters(), lr=params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.80)

    train_dataset = vispr_dataset(data_split='train', shuffle=False, data_percentage=params.data_percentage)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')

    validation_dataset = vispr_dataset(data_split='test', shuffle=False, data_percentage=params.data_percentage)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers)
    print(f'Validation dataset length: {len(validation_dataset)}')
    print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')

    val_array = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45] + [50 + x for x in range(100)]

    print(f'Base learning rate {params.learning_rate}')
    print(f'Scheduler patient {params.lr_patience}')
    print(f'Scheduler drop {params.lr_reduce_factor}')

    lr_flag1 = 0
    lr_counter = 0
    train_loss_best = 1000
    best_score = 0

    for epoch in range(epoch1, params.num_epochs + 1):
        if epoch < params.warmup and lr_flag1 == 0:
            learning_rate = params.warmup_array[epoch]*params.learning_rate/5  # Pretrained drop.
        print()
        print(f'Epoch {epoch} started')
        start = time.time()
        try:
            fb_model, train_loss = train_epoch(epoch, train_dataloader, fa_model, fb_model, params.anon, criterion, optimizer, writer, use_cuda, learning_rate)
            
            if train_loss > train_loss_best:
                lr_counter += 1
            else:
                train_loss_best = train_loss
            if lr_counter > params.lr_patience:
                lr_counter = 0
                learning_rate = learning_rate/params.lr_reduce_factor
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Learning rate dropping to its {params.lr_reduce_factor}th value to {learning_rate} at epoch {epoch}')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            if epoch in val_array:
                pred_dict, label_dict, macro_ap = val_epoch(epoch, validation_dataloader, fa_model, fb_model, params.anon, criterion, use_cuda, writer)

            if macro_ap > best_score:
                best_score = macro_ap
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_loss_{macro_ap:.6f}.pth')
                states = {
                    'epoch': epoch + 1,
                    'lr_counter' : lr_counter,
                    'fb_model_state_dict': fb_model.state_dict(),
                    'pred_dict': pred_dict,
                    'label_dict': label_dict,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)

            # else:
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'lr_counter' : lr_counter,
                'fb_model_state_dict': fb_model.state_dict(),
                'pred_dict': pred_dict,
                'label_dict': label_dict,
                'optimizer': optimizer.state_dict()
                }
            torch.save(states, save_file_path)
            # scheduler.step()
        except:
            print(f'Epoch {epoch} failed.')
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        if learning_rate < 1e-12:
            print(f'Learning rate is very low now, stopping the training.')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default='default_privacy', help='run_id')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default=None, help='run_id')
    parser.add_argument("--arch", dest='arch', type=str, required=False, default='r50', help='model architecture')

    args = parser.parse_args()
    print(f'Run ID: {args.run_id}')

    train_classifier(args.run_id, args.saved_model, args.arch)
