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
import numpy as np
import pandas as pd
import librosa
import soundfile
import pickle
import torch
import torchvision.transforms as transforms
from train_utils.scale_and_cropping import scale, random_cropping
from train_utils.specaugment import time_warp, freq_mask, time_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type=None, data_dict=None, spec_augment_apply=None,
                 scale_apply=None, random_cropping_apply=None, cropping_length=None): 
        self.data_type = data_type
        self.data_x, self.data_y = data_dict['feat'], data_dict['label']
        self.spec_augment_apply = spec_augment_apply
        self.scale_apply = scale_apply
        self.random_cropping_apply = random_cropping_apply
        self.cropping_length = cropping_length
        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transform_scale = transforms.Compose([
            lambda x: scale(x)
        ])
        self.transform_random_cropping = transforms.Compose([
            lambda x: random_cropping(x, self.cropping_length)
        ])
        self.transform_spec_augment = transforms.Compose([
            lambda x: time_warp(x, W=5),
            lambda x: freq_mask(x, num_masks=1),
            lambda x: time_mask(x, num_masks=1)
        ])
        self.length = len(self.data_x)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x, y = self.data_x[idx], self.data_y[idx]
        if self.data_type == 'train':
            if self.scale_apply:
                x = self.transform_scale(x)
            if self.random_cropping_apply:
                x = self.transform_random_cropping(x)
            x = self.transform_to_tensor(x)
            if self.spec_augment_apply:
                x = self.transform_spec_augment(x)
            y = torch.LongTensor(y)
        else:
            if self.scale_apply:
                x = self.transform_scale(x)
            x = self.transform_to_tensor(x)
            y = torch.LongTensor(y)
        return x, y 
        
def deltas(x):
    y = (x[:,:,2:,:] - x[:,:,:-2,:])/10.0
    y = y[:,:,1:-1,:] + (x[:,:,4:,:] - x[:,:,:-4,:])/5.0
    return y     

def data_setup(mode, data_path, train_csv, valid_csv, train_feat, valid_feat, train_divide, 
               sampling_rate, num_audio_channels, sample_duration, num_freq_bins, num_fft_points, 
               hop_length, num_time_bins, norm_apply, deltas_apply, valid_padding_apply): 
    if not os.path.exists('features'):
        os.mkdir('features')
    
    # set train dataset
    train_df = pd.read_csv(train_csv, sep='\t', encoding='ASCII') #','
    train_wavpaths = train_df['filename'].tolist() #filename
    train_labels = train_df['scene_label'].astype('category').cat.codes.values
    class_names = np.unique(train_df['scene_label'])
    num_classes = len(class_names)
    train_y = np.eye(num_classes, dtype='int')[train_labels]
    print (list(class_names))
    print (num_classes)
    
    # extract train feature
    train_dict = {}
    if os.path.exists(train_feat + '.pickle'):
        with open(train_feat + '.pickle', 'rb') as f:
            train_dict = pickle.load(f)
        print ("training data load from: {}.pickle".format(train_feat))
    else:
        if mode == 'dev' and train_divide > 1:
            train_size = int(len(train_wavpaths)/train_divide)
            train_idx = np.random.choice(range(len(train_wavpaths)), int(len(train_wavpaths)/train_divide), replace=False)
            valid_idx = np.random.choice(range(len(valid_wavpaths)), int(train_size/4), replace=False)
            train_wavpaths = np.array(train_wavpaths)[train_idx]
            valid_wavpaths = np.array(valid_wavpaths)[valid_idx]
            train_y = train_y[train_idx]
            valid_y = valid_y[valid_dix]
            num_epochs = 30
        
        train_x = np.zeros((len(train_wavpaths), num_freq_bins, num_time_bins, num_audio_channels), 'float32')
        for i in range(len(train_wavpaths)):
            sig, fs = soundfile.read(data_path + train_wavpaths[i])
            #sig, fs = soundfile.read(data_path + 'etri2019_v2/16kHz+noise/' + train_wavpaths[i])
            
            # cropping and padding
            max_len = sample_duration*sampling_rate
            sig_len = len(sig)
            if sig_len >= max_len:
                sig = sig[:max_len]
            else:
                num_repeats = round((max_len/sig_len) + 1)
                sig_repeat = sig.repeat(num_repeats, 1)
                padded_sig = sig_repeat[:max_len]
                sig = padded_sig
            
            # mono-channel
            if sig[0].shape == (2,):
                sig_mono = []
                sig_mono = (sig[:,0] + sig[:,1]) / 2
                sig = sig_mono
            
            # sampling rate
            if fs != sampling_rate:
                sig = librosa.resample(sig, fs, sampling_rate)

                    
            for ch in range(num_audio_channels):
                if len(sig.shape) == 1:
                    sig = np.expand_dims(sig, -1)
                train_x[i,:,:,ch] = librosa.feature.melspectrogram(sig[:,ch],
                                                                   sr = sampling_rate,
                                                                   n_fft = num_fft_points,
                                                                   hop_length = hop_length,
                                                                   n_mels = num_freq_bins,
                                                                   fmin = 0.0,
                                                                   fmax = sampling_rate/2,
                                                                   htk = True,
                                                                   norm = None)
                train_x[i,:,:,ch] = np.log(train_x[i,:,:,ch] + 1e-8)
                
                if norm_apply:        
                    #mean = np.mean(train_x[i,:,:,ch])
                    #std = np.std(train_x[i,:,:,ch])
                    #train_x[i,:,:,ch] = (train_x[i,:,:,ch] - mean) / std
                    #train_x[i,:,:,ch][np.isnan(train_x[i,:,:,ch])] = 0.0
                    
                    for j in range(len(train_x[i,:,:,ch][:,0])):
                        mean = np.mean(train_x[i,:,:,ch][j,:])
                        std = np.std(train_x[i,:,:,ch][j,:])
                        train_x[i,:,:,ch][j,:] = ((train_x[i,:,:,ch][j,:]-mean)/std)
                        train_x[i,:,:,ch][np.isnan(train_x[i,:,:,ch])]=0.
                    
                    #for j in range(len(train_x[i,:,:,ch][0,:])):
                    #    mean = np.mean(train_x[i,:,:,ch][:,j])
                    #    std = np.std(train_x[i,:,:,ch][:,j])
                    #    train_x[i,:,:,ch][:,j] = ((train_x[i,:,:,ch][:,j]-mean)/std) 
                    #    train_x[i,:,:,ch][np.isnan(train_x[i,:,:,ch])]=0.
                
            if i%1500 == 1499:
                print ("{}/{} training samples done".format(i+1, len(train_wavpaths)))
        
        print ("training data dimension without detla: {}".format(train_x.shape))
        if deltas_apply:
            train_x_deltas = deltas(train_x)
            train_x_deltas_deltas = deltas(train_x_deltas)
            train_x = np.concatenate((train_x[:,:,4:-4,:], train_x_deltas[:,:,2:-2,:], train_x_deltas_deltas), axis=-1)
            num_audio_channels *= 3
        
        train_dict = {'feat':train_x, 'label':train_y}
        with open(train_feat + '.pickle', 'wb') as f:
            pickle.dump(train_dict, f, protocol=4)
        print ("training data dimension: {}".format(train_x.shape))
        print ("training labels dimension: {}".format(train_y.shape))
        
    valid_dict = {}
    # set valid dataset
    if mode == 'dev':
        valid_df = pd.read_csv(valid_csv, sep='\t', encoding='ASCII') #','
        valid_wavpaths = valid_df['filename'].tolist() #filename
        valid_labels = valid_df['scene_label'].astype('category').cat.codes.values
        valid_y = np.eye(num_classes ,dtype='int')[valid_labels]
        
        # extract valid feature
        if os.path.exists(valid_feat + '.pickle'):
            with open(valid_feat + '.pickle', 'rb') as f:
                valid_dict = pickle.load(f)
            print ("validation data load from: {}.pickle".format(valid_feat))
        else:
            valid_x = np.zeros((len(valid_wavpaths), num_freq_bins, num_time_bins, num_audio_channels), 'float32')
            for i in range(len(valid_wavpaths)):
                sig, fs = soundfile.read(data_path + valid_wavpaths[i])
                #sig, fs = soundfile.read(data_path + 'youtube2020_eval_v2/raw/' + valid_wavpaths[i])
                
                max_len = sample_duration*sampling_rate
                sig_len = len(sig)
                if sig_len >= max_len:
                    sig = sig[:max_len]
                else:
                    num_repeats = round((max_len/sig_len) + 1)
                    sig_repeat = sig.repeat(num_repeats, 1)
                    padded_sig = sig_repeat[:max_len]
                    sig = padded_sig
                
                if sig[0].shape == (2,):
                    sig_mono = []
                    sig_mono = (sig[:,0] + sig[:,1]) / 2
                    sig = sig_mono

                if fs != sampling_rate:
                    sig = librosa.resample(sig, fs, sampling_rate)
            
                for ch in range(num_audio_channels):
                    if len(sig.shape) == 1:
                        sig = np.expand_dims(sig, -1)
                    valid_x[i,:,:,ch] = librosa.feature.melspectrogram(sig[:,ch],
                                                                       sr = sampling_rate,
                                                                       n_fft = num_fft_points,
                                                                       hop_length = hop_length,
                                                                       n_mels = num_freq_bins,
                                                                       fmin = 0.0,
                                                                       fmax = sampling_rate/2,
                                                                       htk = True,
                                                                       norm = None)
                    valid_x[i,:,:,ch] = np.log(valid_x[i,:,:,ch] + 1e-8)
                
                    if norm_apply:
                        #mean = np.mean(valid_x[i,:,:,ch])
                        #std = np.std(valid_x[i,:,:,ch])
                        #valid_x[i,:,:,ch] = (valid_x[i,:,:,ch] - mean) / std
                        #valid_x[i,:,:,ch][np.isnan(valid_x[i,:,:,ch])] = 0.0
                        
                        for j in range(len(valid_x[i,:,:,ch][:,0])):
                            mean = np.mean(valid_x[i,:,:,ch][j,:])
                            std = np.std(valid_x[i,:,:,ch][j,:])
                            valid_x[i,:,:,ch][j,:] = ((valid_x[i,:,:,ch][j,:]-mean)/std)
                            valid_x[i,:,:,ch][np.isnan(valid_x[i,:,:,ch])]=0.
                            
                        #for j in range(len(valid_x[i,:,:,ch][0,:])):
                        #    mean = np.mean(valid_x[i,:,:,ch][:,j])
                        #    std = np.std(valid_x[i,:,:,ch][:,j])
                        #    valid_x[i,:,:,ch][:,j] = ((valid_x[i,:,:,ch][:,j]-mean)/std) 
                        #    valid_x[i,:,:,ch][np.isnan(valid_x[i,:,:,ch])]=0.
                
                if i%700 == 699:
                    print ("{}/{} valid samples done".format(i+1, len(valid_wavpaths)))
            
            print ("validation data dimension without detla: {}".format(valid_x.shape))
            if deltas_apply:
                valid_x_deltas = deltas(valid_x)
                valid_x_deltas_deltas = deltas(valid_x_deltas)
                valid_x = np.concatenate((valid_x[:,:,4:-4,:], valid_x_deltas[:,:,2:-2,:], valid_x_deltas_deltas), axis=-1)
                num_audio_channels *= 3
            
            # checking
            if valid_padding_apply:
                pre_padding_length = valid_x.shape[2]
                if np.mod(pre_padding_length, 8) != 0:
                    pad_size = 8 - np.mod(pre_padding_length, 8)
                    temp = np.tile(valid_x[:,:,-1,:], pad_size)
                    temp = np.reshape(temp, (valid_x.shape[0], valid_x.shape[1], -1, valid_x.shape[-1]))
                    valid_x = np.concatenate((valid_x,temp), axis=2)
        
            valid_dict = {'feat':valid_x, 'label':valid_y}
            with open(valid_feat + '.pickle', 'wb') as f:
                pickle.dump(valid_dict, f, protocol=4)
            print ("validation data dimension: {}".format(valid_x.shape))   
            print ("validation labels dimension: {}".format(valid_y.shape))
    return train_dict, valid_dict, num_classes

def eval_data_setup(data_path, eval_csv, eval_feat,
                    sampling_rate, num_audio_channels, sample_duration, num_freq_bins, num_fft_points, 
                    hop_length, num_time_bins, norm_apply, deltas_apply): 
    if not os.path.exists('features'):
        os.mkdir('features')
    
    eval_dict = {}
    # set eval dataset
    eval_df = pd.read_csv(eval_csv, sep='\t', encoding='ASCII') #','
    eval_wavpaths = eval_df['filename'].tolist() #filename
    eval_labels = eval_df['scene_label'].astype('category').cat.codes.values
    class_names = np.unique(eval_df['scene_label'])
    num_classes = len(class_names)
    eval_y = np.eye(num_classes ,dtype='int')[eval_labels]
        
    # extract valid feature
    if os.path.exists(eval_feat + '.pickle'):
        with open(eval_feat + '.pickle', 'rb') as f:
            eval_dict = pickle.load(f)
        print ("evaluation data load from: {}.pickle".format(eval_feat))
    else:
        eval_x = np.zeros((len(eval_wavpaths), num_freq_bins, num_time_bins, num_audio_channels), 'float32')
        for i in range(len(eval_wavpaths)):
            sig, fs = soundfile.read(data_path + eval_wavpaths[i])
            #sig, fs = soundfile.read(data_path + 'youtube2020_eval_v2/raw/' + valid_wavpaths[i])
                
            max_len = sample_duration*sampling_rate
            sig_len = len(sig)
            if sig_len >= max_len:
                sig = sig[:max_len]
            else:
                num_repeats = round((max_len/sig_len) + 1)
                sig_repeat = sig.repeat(num_repeats, 1)
                padded_sig = sig_repeat[:max_len]
                sig = padded_sig
                
            if sig[0].shape == (2,):
                sig_mono = []
                sig_mono = (sig[:,0] + sig[:,1]) / 2
                sig = sig_mono

            if fs != sampling_rate:
                sig = librosa.resample(sig, fs, sampling_rate)
            
            for ch in range(num_audio_channels):
                if len(sig.shape) == 1:
                    sig = np.expand_dims(sig, -1)
                eval_x[i,:,:,ch] = librosa.feature.melspectrogram(sig[:,ch],
                                                                  sr = sampling_rate,
                                                                  n_fft = num_fft_points,
                                                                  hop_length = hop_length,
                                                                  n_mels = num_freq_bins,
                                                                  fmin = 0.0,
                                                                  fmax = sampling_rate/2,
                                                                  htk = True,
                                                                  norm = None)
                eval_x[i,:,:,ch] = np.log(eval_x[i,:,:,ch] + 1e-8)
                
                if norm_apply:  
                    for j in range(len(eval_x[i,:,:,ch][:,0])):
                        mean = np.mean(eval_x[i,:,:,ch][j,:])
                        std = np.std(eval_x[i,:,:,ch][j,:])
                        eval_x[i,:,:,ch][j,:] = ((eval_x[i,:,:,ch][j,:]-mean)/std)
                        eval_x[i,:,:,ch][np.isnan(eval_x[i,:,:,ch])]=0.
                
            if i%700 == 699:
                 print ("{}/{} eval samples done".format(i+1, len(eval_wavpaths)))
            
        print ("evaluation data dimension without detla: {}".format(eval_x.shape))
        if deltas_apply:
            eval_x_deltas = deltas(eval_x)
            eval_x_deltas_deltas = deltas(eval_x_deltas)
            eval_x = np.concatenate((eval_x[:,:,4:-4,:], eval_x_deltas[:,:,2:-2,:], eval_x_deltas_deltas), axis=-1)
            num_audio_channels *= 3
        
        eval_dict = {'feat':eval_x, 'label':eval_y}
        with open(eval_feat + '.pickle', 'wb') as f:
            pickle.dump(eval_dict, f, protocol=4)
        print ("evaluation data dimension: {}".format(eval_x.shape))   
        print ("evaluation labels dimension: {}".format(eval_y.shape))
    return eval_dict, eval_labels, class_names, num_classes