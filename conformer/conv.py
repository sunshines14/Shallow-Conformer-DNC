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
from conformer.activation import Swish, GLU
from conformer.modules import LayerNorm, Transpose


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 0,
            bias = False):
        
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias)

    def forward(self, inputs):
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride = 1,
            padding = 0,
            bias = True):
        
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias)

    def forward(self, inputs):
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_size = 31,
            expansion_factor = 2,
            dropout_p = 0.1,
            device = 'cuda'):
        
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.device = device
        self.sequential = nn.Sequential(
            LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p))

    def forward(self, inputs):
        return self.sequential(inputs.to(self.device)).transpose(1, 2)


class Conv2dSubampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU())

    def forward(self, inputs):
        outputs = self.sequential(inputs)
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        return outputs
    