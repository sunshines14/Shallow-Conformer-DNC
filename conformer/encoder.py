# Copyright (c) 2021, Soohwan Kim (origin). 
#               2021, Soonshin Seo (modification).
#
# All rights reserved.
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
from torch import Tensor
from conformer.feed_forward import FeedForwardNet
from conformer.attention import MultiHeadedSelfAttentionModule
from conformer.conv import ConformerConvModule, Conv2dSubampling
from conformer.modules import ResidualConnectionModule, LayerNorm, Linear


class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim = 512,
            num_attention_heads = 8,
            feed_forward_expansion_factor = 4,
            conv_expansion_factor = 2,
            feed_forward_dropout_p = 0.1,
            attention_dropout_p  = 0.1,
            conv_dropout_p = 0.1,
            conv_kernel_size  = 31,
            half_step_residual = True,
            device = 'cuda'):
        
        super(ConformerBlock, self).__init__()
        self.device = device
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardNet(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                    device=device),
                module_factor=self.feed_forward_residual_factor),
            
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p)),
            
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p)),
            
            ResidualConnectionModule(
                module=FeedForwardNet(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p),
                module_factor=self.feed_forward_residual_factor),
            
            LayerNorm(encoder_dim))

    def forward(self, inputs):
        return self.sequential(inputs.to(self.device))


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim = 80,
            encoder_dim = 512,
            num_layers = 17,
            num_attention_heads = 8,
            feed_forward_expansion_factor = 4,
            conv_expansion_factor = 2,
            input_dropout_p = 0.1,
            feed_forward_dropout_p = 0.1,
            attention_dropout_p = 0.1,
            conv_dropout_p = 0.1,
            conv_kernel_size = 31,
            half_step_residual = True,
            device = 'cuda'):
        
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p))
        
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            device=device
        ).to(device) for _ in range(num_layers)])

    def forward(self, inputs):
        outputs = self.conv_subsample(inputs)
        outputs = self.input_projection(outputs)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs