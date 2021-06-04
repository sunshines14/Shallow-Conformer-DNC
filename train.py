# Copyright (c) 2021, Soonshin Seo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import math
import random
import datetime
import subprocess
import numpy as np
import pandas as pd
import librosa
import soundfile
import torch
import torch.nn as nn
import data_prep
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from train_utils import mixup, optimizers, schedulers

def run_train(train_loader, model, device, criterion, optim, mixup_apply, mixup_alpha, scheduler):
    train_loss = 0.
    correct = 0.
    total = 0.
    
    if device == 'cuda':
        model = nn.DataParallel(model).train()
    else:
        model.train()
        
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), torch.max(targets, 1)[1].to(device)
        
        if mixup_apply:
            inputs, targets_a, targets_b, lam = mixup.mixup_data(inputs, targets, mixup_alpha, device)
        
            optim.zero_grad()
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)

            loss_func = mixup.mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
            loss.backward()
            optim.step()
            scheduler.step()

            total += targets.size(0)
            train_loss += loss.item()
            correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()
        else:
            optim.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)
            loss = criterion(outputs, targets)

            loss.backward()
            optim.step()
            scheduler.step()
        
            total += targets.size(0)
            train_loss += loss.item()
            correct += predicted.eq(targets.data).cpu().sum()
        
        if batch_idx % 10 == 0:
            sys.stdout.write('\r \t {:.5f} {:.5f}'.format(100. * correct/total, train_loss/(batch_idx+1)))
    return (train_loss/batch_idx, 100. * correct/total)

def run_valid(valid_loader, model, device, criterion, epoch, start_epoch, ckpt_flag):
    global best_acc
    valid_loss = 0.
    correct = 0.
    total = 0.
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if epoch == start_epoch: best_acc = 0 
            inputs, targets = inputs.to(device), torch.max(targets, 1)[1].to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)
            loss = criterion(outputs, targets)

            total += targets.size(0)
            valid_loss += loss.item()
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct/total
        if acc > best_acc:
            best_acc = acc
            ckpt_flag = True    
    return (valid_loss/batch_idx, 100. * correct/total, ckpt_flag)

        
def run_epoch(mode, logdir, tag, train_loader, valid_loader, start_epoch, num_epochs, 
              model, device, criterion, optim, mixup_apply, mixup_alpha, scheduler, savedir):
    
    scheduler_epoch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', factor=0.1, patience=10, min_lr=1e-6, verbose=True)
    
    start_epoch = start_epoch
    ckpt_flag = False
    #writer = SummaryWriter(logdir)
    
    valid_losses = []
    valid_accs = []
    if mode == 'dev':
        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_loss, train_acc = run_train(train_loader, model, device, criterion, 
                                              optim, mixup_apply, mixup_alpha, scheduler)
            valid_loss, valid_acc, ckpt_flag = run_valid(valid_loader, model, device, criterion, 
                                                         epoch, start_epoch, ckpt_flag)
            
            #writer.add_scalar('train_acc', train_acc, epoch)
            #writer.add_scalar('train_loss', train_loss, epoch)
            #writer.add_scalar('valid_acc', valid_acc, epoch)
            #writer.add_scalar('valid_loss', valid_loss, epoch)
            
            print('\n{} - train acc: {:.5f} - valid acc: {:.5f} - train loss: {:.5f} - valid loss: {:.5f}'.format(
                epoch, train_acc, valid_acc, train_loss, valid_loss))
            
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            scheduler_epoch.step(valid_loss)
            
            if ckpt_flag:
                torch.save(model.state_dict(), os.path.join(savedir, 'epoch_{}_{:.2f}.pth'.format(epoch, valid_acc)))
                print ("save the model at {}".format(savedir))
                ckpt_flag = False
                
        minposs = valid_losses.index(min(valid_losses))+1
        print('lowest valid loss at epoch is {}'.format(minposs))
        maxposs = valid_accs.index(max(valid_accs))+1
        print('highist valid acc at epoch is {}'.format(maxposs))
    else:
        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_loss, train_acc = run_train(train_loader, model, device, criterion, optim, mixup_apply, mixup_alpha)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            print('\n{} - train acc: {:.5f} - train loss: {:.5f}'.format(
                epoch, train_acc, train_loss))
            scheduler.step()
            torch.save(model.state_dict(), os.path.join(savedir, 'epoch_{}_{}.pth'.format(epoch, train_acc)))
        
def main():
    #random_seed = 100
    #random.seed(random_seed)
    #np.random.seed(random_seed)
    #torch.manual_seed(random_seed)
    #torch.cuda.manual_seed(random_seed)
    #torch.cuda.manual_seed_all(random_seed)
    
    # ==================================================================================== #
    task = '1a'
    mode = 'dev'
    
    if task == '1a':
        data_path = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'
        #data_path = '/home/soonshin/sss/sed/CORPUS/'
        
        if mode == 'dev':
            train_csv = data_path + 'evaluation_setup/fold1_train.csv'
            valid_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
            train_feat = 'features/dcase2020_train_raw_norm'
            valid_feat = 'features/dcase2020_valid_raw_norm'
            
            #train_csv = data_path + 'etri2019_v2/meta/etri2019_train_noise.csv'
            #valid_csv = data_path + 'youtube2020_eval_v2/meta/youtube2020_eval_raw.csv'
            #train_feat = 'features/etri2020_train_noise'
            #valid_feat = 'features/etri2020_valid_noise'
            
        elif mode == 'sub':
            train_csv = data_path + 'evaluation_setup/fold1_train.csv'
            valid_csv = None
            train_feat = 'features/dcase2020_train_raw'
            valid_feat = None
    
    # ==================================================================================== #
    # use in the feature extraction
    train_divide = 1
    sampling_rate = 44100 # 16000
    num_audio_channels = 1 
    sample_duration = 10 # 0.4
    num_freq_bins = 128 # 40
    num_fft_points = 2048 # 321
    hop_length = int(num_fft_points/2)
    num_time_bins = int(np.ceil(sample_duration*sampling_rate / hop_length))
    norm_apply = True # False
    deltas_apply = False # Not yet applied --> 3 channel issue in Conformer module
    valid_padding_apply = False # Not yet applied --> overlab with another padding

    # ==================================================================================== #
    spec_augment_apply = True # num_masks = 1
    mixup_apply = True
    mixup_alpha = 0.2
    scale_apply = False
    random_cropping_apply = False
    cropping_length = 0
    
    batch_size = 512
    num_epochs = 200
    num_workers = 10
    
    optim_type = 'adam' # [adam, adamw, radam]
    lr = 1e-02 # 1e-6
    lr_wd = 1e-6 # 1e-6
    
    lr_betas = (0.9, 0.98)
    lr_eps = 1e-09
    
    scheduler_type = 'warmup'
    peak_lr = 0.05 / math.sqrt(512)
    final_lr = 1e-07
    final_lr_scale = 0.001
    decay_steps = 10000
    warmup_steps = 800
    
    encoder_dim = 16
    num_layers = 2
    num_attention_heads = 4
    feed_forward_expansion_factor = 4
    conv_expansion_factor = 2 
    dropout_p = 0.1
    conv_kernel_size = 7
    
    dnc_input_size = 16
    dnc_hidden_size = 16
    dnc_rnn_type = 'gru'
    dnc_num_layers = 1
    dnc_nr_cells = 16
    dnc_cell_size = 16
    dnc_read_heads = 4
    dnc_gpu_id = 0
    
    split_apply = False
    extractor_apply = False
    
    resume_mode = False
    resume_model_save_path = '2102032113_1a_dev_500'
    resume_num_checkpoint = 290
    
    if torch.cuda.is_available():
        device = 'cuda'
        pin_memory = True
    else:
        device = 'cpu'
        pin_memory = False 
        
    # ==================================================================================== #
    train_dict, valid_dict, num_classes = data_prep.data_setup(mode, data_path, train_csv, valid_csv, train_feat, valid_feat, train_divide, 
                                                               sampling_rate, num_audio_channels, sample_duration, num_freq_bins, num_fft_points, 
                                                               hop_length, num_time_bins, norm_apply, deltas_apply, valid_padding_apply)
    
    train_dataset = data_prep.Dataset('train', 
                                      train_dict,
                                      spec_augment_apply = spec_augment_apply,
                                      scale_apply = scale_apply, 
                                      random_cropping_apply = random_cropping_apply,
                                      cropping_length = cropping_length)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size = batch_size, 
                                               shuffle = True, 
                                               num_workers = num_workers, 
                                               pin_memory = pin_memory)
    
    valid_loader = None
    if mode == 'dev':
        valid_dataset = data_prep.Dataset('valid',
                                          valid_dict,
                                          spec_augment_apply = False,
                                          scale_apply = scale_apply,
                                          random_cropping_apply = False,
                                          cropping_length = cropping_length)
        
        valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                   batch_size = batch_size, 
                                                   shuffle = False, 
                                                   drop_last = False,
                                                   num_workers = num_workers, 
                                                   pin_memory = pin_memory)

    # ==================================================================================== #
    if deltas_apply: num_audio_channels *= 3
        
    from networks import conformer
    model = conformer(num_classes = num_classes, 
                      input_dim = num_freq_bins, 
                      encoder_dim = encoder_dim, 
                      num_layers = num_layers, 
                      num_attention_heads = num_attention_heads,
                      feed_forward_expansion_factor = feed_forward_expansion_factor,
                      conv_expansion_factor = conv_expansion_factor,
                      dropout_p = dropout_p,
                      conv_kernel_size = conv_kernel_size,
                      dnc_input_size = dnc_input_size,
                      dnc_hidden_size = dnc_hidden_size,
                      dnc_rnn_type = dnc_rnn_type,
                      dnc_num_layers = dnc_num_layers,
                      dnc_nr_cells = dnc_nr_cells,
                      dnc_cell_size = dnc_cell_size,
                      dnc_read_heads = dnc_read_heads,
                      dnc_gpu_id = dnc_gpu_id,
                      device = device,
                      split_apply = split_apply,
                      extractor_apply = extractor_apply
                     ).to(device)
        
    #summary(model, input_size=(num_audio_channels, num_freq_bins, num_time_bins))
    total_params = sum(param.numel() for param in model.parameters())
    print ('total_params: ', total_params)
    start_epoch = 1
    
    if resume_mode:
        resume_checkpoint_path = 'save/{}/epoch_{}.pth'.format(resume_model_save_path, str(resume_num_checkpoint))
        model.load_state_dict(torch.load(resume_checkpoint_path))
        print('model for resume loaded from: {}'.format(resume_checkpoint_path))
        start_epoch = resume_num_checkpoint + 1
    
    # ==================================================================================== #
    criterion = nn.CrossEntropyLoss()   
    
    if optim_type is 'adam':
        optim = torch.optim.Adam(model.parameters(), 
                                 lr = lr, 
                                 betas = lr_betas, 
                                 eps = lr_eps, 
                                 weight_decay = lr_wd)
    elif optim_type is 'adamw':
        optim = optimizers.AdamW(model.parameters(), 
                                 lr = lr, 
                                 betas = lr_betas, 
                                 eps = lr_eps, 
                                 weight_decay = lr_wd)
    elif optim_type is 'radam':
        optim = optimizers.RAdam(model.parameters(), 
                                 lr = lr, 
                                 betas = lr_betas, 
                                 eps = lr_eps, 
                                 weight_decay = lr_wd)
    
    total_step = len(train_loader) * num_epochs
    if scheduler_type is 'transformer':
        scheduler = schedulers.TransformerLRScheduler(optim, 
                                                      peak_lr = peak_lr, 
                                                      final_lr = final_lr, 
                                                      final_lr_scale = final_lr_scale, 
                                                      warmup_steps = warmup_steps, 
                                                      decay_steps = decay_steps)
    elif scheduler_type is 'warmup':
        scheduler = schedulers.get_cosine_schedule_with_warmup(optim, 
                                                               num_warmup_steps = warmup_steps, 
                                                               num_training_steps = total_step, 
                                                               num_cycles = 1., # 0.5 is default
                                                               last_epoch = -1)
    elif scheduler_type is 'restarts_warmup':
        scheduler= schedulers.get_cosine_with_hard_restarts_schedule_with_warmup(optim, 
                                                                                 num_warmup_steps = warmup_steps, 
                                                                                 num_training_steps = total_step, 
                                                                                 num_cycles = 1., 
                                                                                 last_epoch = -1)
    
    # ==================================================================================== #
    if not os.path.exists('save'):
        os.mkdir('save')
        
    if resume_mode:
        tag = resume_model_save_path + '_resume'
    else:
        stamp = datetime.datetime.now().strftime('%y%m%d%H%M')
        tag = stamp + '_' + task + '_' + mode + '_'+ str(num_epochs)
        
    tmpdir = os.path.join('save', tag)
    savedir = os.path.join(os.getcwd(), tmpdir)
    print ("save path: {}".format(savedir))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    ltmpdir = os.path.join('logs', tag)
    logdir = os.path.join(os.getcwd(), ltmpdir)
    print ("log path: {}".format(logdir))
    
    subprocess.check_call(['cp', 'train.py', savedir])
    subprocess.check_call(['cp', 'networks.py', savedir])
    
    # ==================================================================================== #
    run_epoch(mode, logdir, tag, train_loader, valid_loader, start_epoch, num_epochs, 
              model, device, criterion, optim, mixup_apply, mixup_alpha, scheduler, savedir)
    
    # ==================================================================================== #
    
if __name__ == '__main__':
    main()