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
from conformer.activation import Swish
from conformer.modules import LayerNorm, Linear


class FeedForwardNet(nn.Module):
    def __init__(
            self,
            encoder_dim = 512,
            expansion_factor = 4,
            dropout_p = 0.1,
            device = 'cuda'):
        
        super(FeedForwardNet, self).__init__()
        self.device = device
        self.sequential = nn.Sequential(
            LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p))

    def forward(self, inputs):
        return self.sequential(inputs.to(self.device))