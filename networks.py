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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from conformer.encoder import ConformerEncoder
from conformer.extractor import Extractor
from dnc import DNC


class conformer(nn.Module):
    def __init__(self, 
                 num_classes, 
                 input_dim, 
                 encoder_dim, 
                 num_layers, 
                 num_attention_heads,
                 feed_forward_expansion_factor,
                 conv_expansion_factor,
                 dropout_p,
                 conv_kernel_size,
                 dnc_input_size,
                 dnc_hidden_size,
                 dnc_rnn_type,
                 dnc_num_layers,
                 dnc_nr_cells,
                 dnc_cell_size,
                 dnc_read_heads,
                 dnc_gpu_id,
                 device,
                 split_apply,
                 extractor_apply):
        super(conformer, self).__init__()
        
        # adapted from (https://github.com/sooftware/conformer)
        self.num_classes = num_classes
        self.input_dim  = input_dim
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feed_forward_expansion_factor = feed_forward_expansion_factor
        self.conv_expansion_factor = conv_expansion_factor
        self.input_dropout_p = dropout_p
        self.feed_forward_dropout_p = dropout_p
        self.attention_dropout_p = dropout_p
        self.conv_dropout_p = dropout_p
        self.conv_kernel_size = conv_kernel_size
        self.half_step_residual = True
        
        self.dnc_input_size = dnc_input_size
        self.dnc_hidden_size = dnc_hidden_size
        self.dnc_rnn_type = dnc_rnn_type
        self.dnc_num_layers = dnc_num_layers
        self.dnc_nr_cells = dnc_nr_cells
        self.dnc_cell_size = dnc_cell_size
        self.dnc_read_heads = dnc_read_heads
        self.dnc_gpu_id = dnc_gpu_id
        
        self.device = device
        self.split_apply = split_apply
        self.extractor_apply = extractor_apply
        
        if self.split_apply == True and self.extractor_apply == False:
            self.input_dim  = int(self.input_dim/2)
            
        if self.split_apply == False and self.extractor_apply == True:
            self.input_dim  = int(input_dim*2)
        
        self.encoder = ConformerEncoder(input_dim = self.input_dim,
                                        encoder_dim = self.encoder_dim,
                                        num_layers = self.num_layers,
                                        num_attention_heads = self.num_attention_heads,
                                        feed_forward_expansion_factor = self.feed_forward_expansion_factor,
                                        conv_expansion_factor = self.conv_expansion_factor,
                                        input_dropout_p = self.input_dropout_p,
                                        feed_forward_dropout_p = self.feed_forward_dropout_p,
                                        attention_dropout_p = self.attention_dropout_p,
                                        conv_dropout_p = self.conv_dropout_p,
                                        conv_kernel_size = self.conv_kernel_size,
                                        half_step_residual = self.half_step_residual,
                                        device = self.device
                                       ).to(device)
        
        self.dnc = DNC(input_size = dnc_input_size,
                       hidden_size = dnc_hidden_size,
                       rnn_type = dnc_rnn_type,
                       num_layers = dnc_num_layers,
                       nr_cells = dnc_nr_cells,
                       cell_size = dnc_cell_size,
                       read_heads = dnc_read_heads,
                       batch_first = True,
                       gpu_id = dnc_gpu_id
                      ).to(device)
        
        self.gap = nn.AdaptiveAvgPool2d((1, self.encoder_dim))
        self.fc = nn.Linear(self.encoder_dim, self.num_classes)
        
        #self.extractor = Extractor(
        #    n_in_channel = 1,
        #    activation = "glu", 
        #    conv_dropout = 0.5,
        #    kernel_size = 7 * [3], 
        #    padding = 7 * [1], 
        #    stride = 7 * [1], 
        #    nb_filters = [16, 32, 64, 128, 128, 128, 128],
        #    pooling = [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 1]]
        #).to(device)
        
    def freq_split1(self, x):
        return x[:,:,:,0:64]

    def freq_split2(self, x):
        return x[:,:,:,64:128]
    
    def forward(self, inputs):
        print (inputs.size())
        exit (0)
        outputs = inputs.permute(0, 1, 3, 2)
        
        if self.split_apply:
            outputs_splited1 = self.freq_split1(outputs)
            outputs_splited2 = self.freq_split2(outputs)
            
            if self.extractor_apply:
                outputs_splited1 = self.extractor(outputs_splited1)
                outputs_splited2 = self.extractor(outputs_splited2)
                
                bs, ch, frames, freq = outputs_splited1.size()
                outputs_splited1 = outputs_splited1.permute(0, 2, 1, 3)
                outputs_splited1 = outputs_splited1.contiguous().view(bs, frames, ch * freq)
                outputs_splited1 = outputs_splited1.unsqueeze(1)
                
                bs, ch, frames, freq = outputs_splited2.size()
                outputs_splited2 = outputs_splited2.permute(0, 2, 1, 3)
                outputs_splited2 = outputs_splited2.contiguous().view(bs, frames, ch * freq)
                outputs_splited2 = outputs_splited2.unsqueeze(1)
                
            outputs_splited1 = self.encoder(outputs_splited1)
            outputs_splited2 = self.encoder(outputs_splited2)
            outputs_concat = torch.cat((outputs_splited1, outputs_splited2), 1)
            outputs, _ = self.dnc(outputs, (None, None, None), reset_experience=True, pass_through_memory=True)
            
            outputs = self.gap(outputs_concat)
            outputs = outputs.squeeze(1)
            outputs = self.fc(outputs).log_softmax(dim=-1)
        else:
            if self.extractor_apply:
                outputs = self.extractor(outputs)
                
                bs, ch, frames, freq = outputs.size()
                outputs = outputs.permute(0, 2, 1, 3)
                outputs = outputs.contiguous().view(bs, frames, ch * freq)
                outputs = outputs.unsqueeze(1)
                
            outputs = self.encoder(outputs)
            outputs, _ = self.dnc(outputs, (None, None, None), reset_experience=True, pass_through_memory=True)
            outputs = self.gap(outputs)
            outputs = outputs.squeeze(1)
            outputs = self.fc(outputs).log_softmax(dim=-1)
        return outputs