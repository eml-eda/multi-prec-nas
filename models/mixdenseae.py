#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso <matteo.risso@polito.it>                             *
#*----------------------------------------------------------------------------*

import math
import numpy as np
import torch.nn as nn

from . import quant_module as qm

# MR
__all__ = [
    'mixdenseae_w0248a8_multiprec', 
    'mixdenseae_w248a8_multiprec', 'mixdenseae_w248a8_chan',
    'mixdenseae_w248a248_multiprec', 'mixdenseae_w248a248_chan',
]

# MR
def fc(conv_func, in_planes, out_planes, stride=1, groups=1, search_fc=None, **kwargs):
    "fc mapped to conv"
    return conv_func(in_planes, out_planes, kernel_size=1, groups=groups, stride=stride,
                     padding=0, bias=False, fc=search_fc, **kwargs)

# MR
class TinyMLDenseAe(nn.Module):
    def __init__(self, conv_func, search_fc=None, num_classes=640, input_size=640,
                 bnaff=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if search_fc in self.search_types:
            self.search_fc = search_fc
        else:
            self.search_fc = False
        super().__init__()
        self.gumbel = kwargs.get('gumbel', False)
        
        self.fc_0 = fc(conv_func, input_size, 128, search_fc=self.search_fc, fix_qtz=True, **kwargs)
        self.bn_0 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_1 = fc(conv_func, 128, 128, search_fc=self.search_fc, **kwargs)
        self.bn_1 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_2 = fc(conv_func, 128, 128, search_fc=self.search_fc, **kwargs)
        self.bn_2 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_3 = fc(conv_func, 128, 128, search_fc=self.search_fc, **kwargs)
        self.bn_3 = nn.BatchNorm2d(128, affine=bnaff)

        self.fc_4 = fc(conv_func, 128, 8, search_fc=self.search_fc, **kwargs)
        self.bn_4 = nn.BatchNorm2d(8, affine=bnaff)

        self.fc_5 = fc(conv_func, 8, 128, search_fc=self.search_fc, **kwargs)
        self.bn_5 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_6 = fc(conv_func, 128, 128, search_fc=self.search_fc, **kwargs)
        self.bn_6 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_7 = fc(conv_func, 128, 128, search_fc=self.search_fc, **kwargs)
        self.bn_7 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_8 = fc(conv_func, 128, 128, search_fc=self.search_fc, **kwargs)
        self.bn_8 = nn.BatchNorm2d(128, affine=bnaff)

        self.fc_9 = fc(conv_func, 128, num_classes, search_fc=self.search_fc, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, temp, is_hard):
        w_complexity = 0

        x, w_comp = self.fc_0(x, temp, is_hard) # 640 -> 128
        w_complexity += w_comp
        x = self.bn_0(x)

        x, w_comp = self.fc_1(x, temp, is_hard) # 128 -> 128
        w_complexity += w_comp
        x = self.bn_1(x)

        x, w_comp = self.fc_2(x, temp, is_hard) # 128 -> 128
        w_complexity += w_comp
        x = self.bn_2(x)

        x, w_comp = self.fc_3(x, temp, is_hard) # 128 -> 128
        w_complexity += w_comp
        x = self.bn_3(x)

        x, w_comp = self.fc_4(x, temp, is_hard) # 128 -> 8
        w_complexity += w_comp
        x = self.bn_4(x)

        # 8 -> 128
        x, w_comp = self.fc_5(x, temp, is_hard)
        w_complexity += w_comp
        x = self.bn_5(x)

        x, w_comp = self.fc_6(x, temp, is_hard)
        w_complexity += w_comp
        x = self.bn_6(x)

        x, w_comp = self.fc_7(x, temp, is_hard)
        w_complexity += w_comp
        x = self.bn_7(x)

        x, w_comp = self.fc_8(x, temp, is_hard)
        w_complexity += w_comp
        x = self.bn_8(x)

        # 128 -> 640
        x, w_comp = self.fc_9(x, temp, is_hard)
        w_complexity += w_comp

        return x[:, :, 0, 0], w_complexity

    def complexity_loss(self):
        #size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss_cycle()
                #size_product += [m.size_product]
        #normalizer = size_product[0].item()
        #loss /= normalizer
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw

# MR
def mixdenseae_w0248a8_multiprec(**kwargs):
    return TinyMLDenseAe(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[0, 2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixdenseae_w248a8_multiprec(**kwargs):
    return TinyMLDenseAe(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixdenseae_w248a8_chan(**kwargs):
    return TinyMLDenseAe(qm.MixActivChanConv2d, search_fc='mixed', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixdenseae_w248a248_multiprec(**kwargs):
    return TinyMLDenseAe(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)

# MR
def mixdenseae_w248a248_chan(**kwargs):
    return TinyMLDenseAe(qm.MixActivChanConv2d, search_fc='mixed', wbits=[2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)
