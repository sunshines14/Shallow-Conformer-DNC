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

import torch
import random
import numpy as np
import sklearn.metrics as metrics
import data_prep
from torchsummary import summary
from torch.autograd import Variable
from plots import plot_confusion_matrix


def run_eval(eval_loader, model, device):
    eval_predicted = []
    correct = 0.
    total = 0.
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), torch.max(targets, 1)[1].to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            eval_predicted.extend(predicted.cpu().detach().numpy())

        acc = 100. * correct/total  
    return acc, eval_predicted

def main():
    # ==================================================================================== #
    random_seed = 100
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # ==================================================================================== #
    task = '1a'
    
    if task == '1a':
        data_path = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'
        eval_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
        eval_feat = 'features/dcase2020_eval_raw_norm_1s'
        cm_comments = '_1s'
        
        eval_model_save_path = 'thin-conformer-S-aug-head4-3s'
        eval_num_checkpoint = 106
        valid_acc = 49.70
    
    # ==================================================================================== #
    # use in the feature extraction
    train_divide = 1
    sampling_rate = 44100
    num_audio_channels = 1 
    sample_duration = 10
    num_freq_bins = 128
    num_fft_points = 2048
    hop_length = int(num_fft_points/2)
    num_time_bins = int(np.ceil(sample_duration*sampling_rate / hop_length))
    norm_apply = True
    deltas_apply = False # Not yet applied --> 3 channel issue in Conformer module

    # ==================================================================================== # 
    scale_apply = False
    
    batch_size = 512
    num_workers = 10
    
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
    
    if torch.cuda.is_available():
        device = 'cuda'
        pin_memory = True
    else:
        device = 'cpu'
        pin_memory = False 
        
    # ==================================================================================== #
    eval_dict, eval_labels, class_names, num_classes = data_prep.eval_data_setup(data_path, eval_csv, eval_feat,
                                                                            sampling_rate, num_audio_channels, sample_duration, num_freq_bins, num_fft_points, 
                                                                            hop_length, num_time_bins, norm_apply, deltas_apply)
    
    eval_loader = None
    eval_dataset = data_prep.Dataset('eval',
                                     eval_dict,
                                     spec_augment_apply = False,
                                     scale_apply = scale_apply,
                                     random_cropping_apply = False,
                                     cropping_length = 0)
        
    eval_loader = torch.utils.data.DataLoader(eval_dataset, 
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
    
    # ==================================================================================== #
    eval_load_path = 'save/{}/epoch_{}_{}.pth'.format(eval_model_save_path, str(eval_num_checkpoint), str(valid_acc))
    model.load_state_dict(torch.load(eval_load_path))
    print('model for evaluation loaded from: {}'.format(eval_load_path))
    
    eval_plot_save_path = 'save/{}/epoch_{}{}.png'.format(eval_model_save_path, str(eval_num_checkpoint), cm_comments)
    
    acc, eval_predicted = run_eval(eval_loader, model, device)
    
    # ==================================================================================== #
    plot_confusion_matrix(eval_labels, eval_predicted, class_names, normalize=True, title=None, png_name=eval_plot_save_path)
    overall_accuracy = metrics.accuracy_score(eval_labels, eval_predicted)
    precision_mat = metrics.precision_score(eval_labels, eval_predicted, average=None, zero_division='warn')
    recall_mat = metrics.recall_score(eval_labels, eval_predicted, average=None, zero_division='warn')
    f1_score_mat = metrics.f1_score(eval_labels, eval_predicted, average=None, zero_division='warn')
    precision = metrics.precision_score(eval_labels, eval_predicted, average='weighted', zero_division='warn')
    recall = metrics.recall_score(eval_labels, eval_predicted, average='weighted', zero_division='warn')
    f1_score = metrics.f1_score(eval_labels, eval_predicted, average='weighted', zero_division='warn')

    print(metrics.classification_report(eval_labels, eval_predicted))
    print(metrics.confusion_matrix(eval_labels, eval_predicted))

    print("per-class precision: ", precision_mat)
    print("per-class recall: ", recall_mat)
    print("per-class f1-score: ", f1_score_mat)

    print("accuracy :", overall_accuracy)
    print("precision :", precision)
    print("recall :", recall)
    print("f1 score :", f1_score)
    
    # ==================================================================================== #
        
if __name__ == '__main__':
    main()