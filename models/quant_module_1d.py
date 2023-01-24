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

from __future__ import print_function
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185}


# DJP: (Check for DW)
class _channel_asym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        ch_max, _ = x.view(x.size(0), -1).max(1)
        ch_min, _ = x.view(x.size(0), -1).min(1)
        return _channel_min_max_quantize_common(x, ch_min, ch_max, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# DJP
def _channel_min_max_quantize_common(x, ch_min, ch_max, bit):
    if bit != 0:
        ch_range = ch_max - ch_min
        ch_range.masked_fill_(ch_range.eq(0), 1)
        n_steps = 2 ** bit - 1
        S_w = ch_range / n_steps
        S_w = S_w.view((x.size(0), 1, 1))
        y = x.div(S_w).round().mul(S_w)
    else:
        y = torch.zeros(x.shape, device=x.device)
    return y


# DJP: (Check for DW)
class _channel_asym_min_max_quantize_linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        ch_max, _ = x.view(x.size(0), -1).max(1)
        ch_min, _ = x.view(x.size(0), -1).min(1)
        return _channel_min_max_quantize_common_linear(x, ch_min, ch_max, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# DJP
def _channel_min_max_quantize_common_linear(x, ch_min, ch_max, bit):
    ch_range = ch_max - ch_min
    ch_range.masked_fill_(ch_range.eq(0), 1)
    n_steps = 2 ** bit - 1
    S_w = ch_range / n_steps
    S_w = S_w.view((x.size(0), 1))
    y = x.div(S_w).round().mul(S_w)
    return y


# MR:
class _bias_asym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        if x.shape[0] == 1 and x < 0:
            return _bias_min_max_quantize_common(x, x.min(), 0, bit)
        elif x.shape[0] == 1 and x >= 0:
            return _bias_min_max_quantize_common(x, 0, x.max(), bit)
        else:
            return _bias_min_max_quantize_common(x, x.min(), x.max(), bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# MR
def _bias_min_max_quantize_common(x, ch_min, ch_max, bit):
    bias_range= ch_max - ch_min
    n_steps = 2 ** bit - 1
    S_bias = bias_range / n_steps
    y = (x / S_bias).round() * S_bias
   
    return y


# DJP
def asymmetric_linear_quantization_scale_factor(num_bits, saturation_min, saturation_max):
    n = 2 ** num_bits - 1
    return n / (saturation_max - saturation_min)


# DJP
def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).floor_()
        return input
    return torch.floor(scale_factor * input)


# DJP
def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


# DJP
def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


# DJP
class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, clip_val.data[0])
        output = clamp(input, 0, clip_val.data[0], inplace)
        output = linear_quantize(output, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.le(0), 0)
        grad_input.masked_fill_(input.ge(clip_val.data[0]), 0)

        grad_alpha = grad_output.clone()
        grad_alpha.masked_fill_(input.lt(clip_val.data[0]), 0)
#        grad_alpha[input.lt(clip_val.data[0])] = 0
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, None, None, None


# DJP (w.r.t Manuele's code I changed inplace to false to avoid error)
class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val=6, dequantize=True, inplace=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = LearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


# DJP
class MixQuantPaCTActiv(nn.Module):

    def __init__(self, bits):
        super(MixQuantPaCTActiv, self).__init__()
        self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(LearnedClippedLinearQuantization(num_bits=bit))

    def forward(self, input):
        outs = []
        # self.alpha_activ = torch.nn.Parameter(clamp(self.alpha_activ,-100,+100))
        sw = F.softmax(self.alpha_activ, dim=0)
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


# DJP
class SharedMixQuantChanConv1d(nn.Module):

    def __init__(self, inplane, outplane, bits, gumbel=False, **kwargs):
        super().__init__()
        #assert not kwargs['bias']
        self.bits = bits
        self.gumbel = gumbel
        kernel_size = kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_init = kwargs.pop('alpha_init', 'same')
        if self.alpha_init == 'same':
            self.alpha_weight.data.fill_(0.01)
        elif self.alpha_init == 'scaled':
            max_prec = max(self.bits)
            for i in range(len(self.bits)):
                self.alpha_weight.data[i].fill_(self.bits[i] / max_prec)
        self.conv = nn.Conv1d(inplane, outplane, **kwargs)

    def forward(self, input, temp, is_hard):
        mix_quant_weight = []
        mix_wbit = 0
        # self.alpha_weight = torch.nn.Parameter(clamp(self.alpha_weight, -100, +100))
        if not self.gumbel:
            sw = F.softmax(self.alpha_weight / temp, dim=0)
        else:
            # If is_hard is True the output is one-hot
            sw = F.gumbel_softmax(self.alpha_weight, tau=temp, hard=is_hard, dim=0)
        conv = self.conv
        weight = conv.weight
        bias = getattr(conv, 'bias', None)
        for i, bit in enumerate(self.bits):
            quant_weight = _channel_asym_min_max_quantize.apply(weight, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
            # Complexity
            mix_wbit += sw[i] * bit
        if bias is not None:
            quant_bias = _bias_asym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv1d(
            input, mix_quant_weight, quant_bias, conv.stride, conv.padding, conv.dilation, conv.groups)
         # Measure weight complexity for reg-loss
        w_complexity = mix_wbit * self.param_size
        return out, w_complexity


# DJP
class SharedMultiPrecConv1d(nn.Module):

    def __init__(self, inplane, outplane, bits, gumbel=False, **kwargs):
        super().__init__()
        self.bits = bits
        self.gumbel = gumbel
        # if True, when argmax is zero the channel is effectively pruned
        # if False, when argmax is zero the effective weight tensor (mix_quant_weight) is used
        self.hard_prune = False
        self.prune = 0 in bits
        self.cout = outplane
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout))
        self.alpha_init = kwargs.pop('alpha_init', 'same')
        if self.alpha_init == 'same' or self.alpha_init is None:
            if self.gumbel:
                val_equiprob = 1.0 / len(self.bits)
                init_logit = math.log(val_equiprob/(1-val_equiprob))
            else:
                init_logit = 0.01
            self.alpha_weight.data.fill_(init_logit)
        elif self.alpha_init == 'scaled':
            max_prec = max(self.bits)
            scaled_val = torch.tensor([bit/max_prec for bit in self.bits])
            if self.gumbel:
                scaled_prob = F.softmax(scaled_val, dim=0)
                scaled_logit = torch.log(scaled_prob/(1-scaled_prob))
            else:
                scaled_logit = scaled_val
            for i in range(len(self.bits)):
                self.alpha_weight.data[i].fill_(scaled_logit[i])
        else:
            raise ValueError(f'Unknown alpha_init: {self.alpha_init}')
        self.conv = nn.Conv1d(inplane, outplane, **kwargs)
        self.register_buffer('sw_buffer', torch.zeros(self.alpha_weight.shape, dtype=torch.float))

    def forward(self, input, temp, is_hard):
        mix_quant_weight = []
        mix_wbit = 0
        if not self.gumbel:
            sw = F.softmax(self.alpha_weight/temp, dim=0)
        else:
            # If is_hard is True the output is one-hot
            if self.training: # If model.train()
                sw = F.gumbel_softmax(self.alpha_weight, tau=temp, hard=is_hard, dim=0)
                self.sw_buffer = sw.clone().detach()
            else: # If model.eval()
                sw = self.sw_buffer
        # Always False for now (old implementationf of hard-pruning)
        if self.prune and self.hard_prune:
            sw_bin = _prune_channels.apply(sw)
        conv = self.conv
        weight = conv.weight
        bias = getattr(conv, 'bias', None)
        for i, bit in enumerate(self.bits):
            quant_weight = _channel_asym_min_max_quantize.apply(weight, bit)
            scaled_quant_weight = quant_weight * sw[i].view((self.cout, 1, 1))
            mix_quant_weight.append(scaled_quant_weight)
            # Complexity
            mix_wbit += sum(sw[i]) * bit
        mix_wbit = mix_wbit / self.cout
        if bias is not None:
            quant_bias = _bias_asym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        if self.prune and self.hard_prune:
            mix_quant_weight = sum(mix_quant_weight) * sw_bin.view((self.cout, 1, 1))
        else:
            mix_quant_weight = sum(mix_quant_weight)
        out = F.conv1d(
            input, mix_quant_weight, quant_bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        # Measure weight complexity for reg-loss
        w_complexity = mix_wbit * self.param_size
        return out, w_complexity


# DJP
class MultiPrecActivConv1d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=True, fc=None, **kwargs):
        super().__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        
        self.reg_target = kwargs.pop('reg_target', 'ops')

        self.first_layer = False
        kwargs.pop('first_layer', None)

        self.fix_qtz = kwargs.pop('fix_qtz', False)

        self.search_types = ['fixed', 'mixed', 'multi']
        if fc in self.search_types:
            self.fc = fc
        else:
            self.fc = False
        self.gumbel = kwargs.pop('gumbel', False)
        self.temp = 1

        # build mix-precision branches
        # TODO: change here for multi-prec activations
        self.mix_activ = MixQuantPaCTActiv(self.abits)
        # for multiprec, only share-weight is feasible
        assert share_weight == True
        if not self.fc:
            self.mix_weight = SharedMultiPrecConv1d(inplane, outplane, self.wbits, gumbel=self.gumbel, **kwargs) 
        else:
            # For the final fc layer the pruning bit-width (i.e., 0) makes no sense
            _wbits = copy.deepcopy(self.wbits)
            if 0 in _wbits:
                _wbits.remove(0)
            # If the layer is fc we can use:
            if self.fc == 'fixed':
                # - Fixed quantization on 8bits
                self.mix_weight = QuantMixChanConv1d(inplane, outplane, 8, **kwargs)
            elif self.fc == 'mixed':
                # - Mixed-precision search 
                self.mix_weight = SharedMixQuantChanConv1d(inplane, outplane, _wbits, gumbel=self.gumbel, **kwargs)
            elif self.fc == 'multi':
                # - Multi-precision search
                self.mix_weight = SharedMultiPrecConv1d(inplane, outplane, _wbits, gumbel=self.gumbel, **kwargs)
            else:
                raise ValueError(f"Unknown fc search, possible values are {self.search_types}")
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        kernel_size = kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input, temp, is_hard):
        self.temp = temp
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.fix_qtz:
            out = self.mix_activ(input)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8)
        out, w_complexity = self.mix_weight(out, temp, is_hard)
        return out, w_complexity

    def complexity_loss(self):
        if not self.first_layer:
            sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
            mix_abit = 0
            abits = self.mix_activ.bits
            for i in range(len(abits)):
                mix_abit += sw[i] * abits[i]
        else:
            mix_abit = 8

        if not self.fc or self.fc == 'multi':
            sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
            mix_wbit = 0
            wbits = self.mix_weight.bits
            cout = self.mix_weight.cout
            for i in range(len(wbits)):
                mix_wbit += sum(sw[i]) * wbits[i]
            mix_wbit = mix_wbit / cout
        else:
            if self.fc == 'fixed':
                mix_wbit = 8
            elif self.fc == 'mixed':
                sw1 = F.softmax(self.mix_weight.alpha_weight, dim=0)
                mix_wbit = 0
                wbits = self.mix_weight.bits
                for i in range(len(wbits)):
                    mix_wbit += sw1[i] * wbits[i]

        if self.reg_target == 'ops':
            complexity = self.size_product.item() * mix_abit * mix_wbit
        elif self.reg_target == 'weights':
            complexity = self.param_size * mix_wbit
        else:
            raise ValueError(f"Unknown regularization target: {self.reg_target}")
        
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        if not self.first_layer:
            prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
            prob_activ = prob_activ.detach().cpu().numpy()
            best_activ = prob_activ.argmax()
            mix_abit = 0
            abits = self.mix_activ.bits
            for i in range(len(abits)):
                mix_abit += prob_activ[i] * abits[i]
        else:
            prob_activ = 1
            mix_abit = 8
        if not self.fc or self.fc == 'multi':
            prob_weight = F.softmax(self.mix_weight.alpha_weight/self.temp, dim=0)
            prob_weight = prob_weight.detach().cpu().numpy()
            best_weight = prob_weight.argmax(axis=0)
            mix_wbit = 0
            wbits = self.mix_weight.bits
            cout = self.mix_weight.cout
            for i in range(len(wbits)):
                mix_wbit += sum(prob_weight[i]) * wbits[i]
            mix_wbit = mix_wbit / cout
        else:
            if self.fc == 'fixed':
                prob_weight = 1
                mix_wbit = 8
            elif self.fc == 'mixed':
                prob_weight = F.softmax(self.mix_weight.alpha_weight/self.temp, dim=0)
                prob_weight = prob_weight.detach().cpu().numpy()
                best_weight = prob_weight.argmax(axis=0)
                mix_wbit = 0
                wbits = self.mix_weight.bits
                for i in range(len(wbits)):
                    mix_wbit += prob_weight[i] * wbits[i]

        weight_shape = list(self.mix_weight.conv.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        if not self.fc or self.fc == 'multi':
            best_wbit = sum([wbits[_] for _ in best_weight]) / cout
        else:
            if self.fc == 'fixed':
                best_wbit = 8
                best_weight = 8
            elif self.fc == 'mixed':
                best_wbit = wbits[best_weight]

        if not self.first_layer:
            best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
            bitops = size_product * abits[best_activ] * best_wbit
            bita = memory_size * abits[best_activ]
        else:
            best_arch = {'best_activ': [8], 'best_weight': [best_weight]}
            bitops = size_product * 8 * best_wbit
            bita = memory_size * 8

        bitw = self.param_size * best_wbit
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit

        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw

# DJP
class MixActivChanConv1d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=True, fc=None, **kwargs):
        super().__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits

        self.reg_target = kwargs.pop('reg_target', 'ops')

        self.first_layer = False
        kwargs.pop('first_layer', None)

        self.fix_qtz = kwargs.pop('fix_qtz', False)

        self.fc = fc
        self.gumbel = kwargs.pop('gumbel', False)
        
        # build mix-precision branches
        self.mix_activ = MixQuantPaCTActiv(self.abits)
        self.share_weight = share_weight
        if share_weight:
            self.mix_weight = SharedMixQuantChanConv1d(inplane, outplane, self.wbits, **kwargs)
        else:
            self.mix_weight = MixQuantChanConv1d(inplane, outplane, self.wbits, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        kernel_size = kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input, temp, is_hard):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.fix_qtz:
            out = self.mix_activ(input)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8) 
        out, w_complexity = self.mix_weight(out, temp, is_hard)
        return out, w_complexity

    def complexity_loss(self):
        if not self.first_layer:
            sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
            mix_abit = 0
            abits = self.mix_activ.bits
            for i in range(len(abits)):
                mix_abit += sw[i] * abits[i]
        else:
            mix_abit = 8
        sw1 = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw1[i] * wbits[i]
        if self.reg_target == 'ops':
            complexity = self.size_product.item() * mix_abit * mix_wbit
        elif self.reg_target == 'weights':
            complexity = self.param_size * mix_wbit
        else:
            raise ValueError(f'Unknown regularization target: {self.reg_target}')
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        if not self.first_layer:
            prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
            prob_activ = prob_activ.detach().cpu().numpy()
            best_activ = prob_activ.argmax()
            mix_abit = 0
            abits = self.mix_activ.bits
            for i in range(len(abits)):
                mix_abit += prob_activ[i] * abits[i]
        else:
            prob_activ = 1
            mix_abit = 8
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        if self.share_weight:
            weight_shape = list(self.mix_weight.conv.weight.shape)
        else:
            weight_shape = list(self.mix_weight.conv_list[0].weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        if not self.first_layer:
            best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
            bitops = size_product * abits[best_activ] * wbits[best_weight]
            bita = memory_size * abits[best_activ]
        else:
            best_arch = {'best_activ': [32], 'best_weight': [best_weight]}
            bitops = size_product * 32 * wbits[best_weight]
            bita = memory_size * 32
        bitw = self.param_size * wbits[best_weight]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw

# DJP
class QuantizedChanConv1d(nn.Module):
    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight = True, first_layer = False, **kwargs):
        super(QuantizedChanConv1d, self).__init__()
        if wbits is None:
            self.wbits = 8
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = 8
        else:
            self.abits = abits
        # build mix-precision branches
        self.first_layer = first_layer
        if self.first_layer != True:
            self.input_act = LearnedClippedLinearQuantization(num_bits= self.abits)
        self.quantized_weight = nn.Conv1d(inplane, outplane, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1

    def forward(self, input):
        if self.first_layer != True:
            input = self.input_act(input)
        conv = self.quantized_weight
        weight = conv.weight
        quant_weight = _channel_asym_min_max_quantize.apply(weight, self.wbits)
        out = F.conv1d(input, quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


# MR
class QuantPaCTActiv(nn.Module):

    def __init__(self, bits):
        super().__init__()
        if type(bits) == int:
            self.bits = [bits]
        else:
            self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)), requires_grad=False)
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(LearnedClippedLinearQuantization(num_bits=bit))

    def forward(self, input):
        outs = []
        # self.alpha_activ = torch.nn.Parameter(clamp(self.alpha_activ,-100,+100))
        sw = F.one_hot(torch.argmax(self.alpha_activ), num_classes = len(self.bits))
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


# MR
class QuantMixActivChanConv1d(nn.Module):

    def __init__(self, inplane, outplane, wbits, abits, fc=False, **kwargs):
        super().__init__()
        self.abits = abits
        self.wbit = wbits
        self.fc = False

        self.first_layer = kwargs.pop('first_layer', False)

        self.mix_activ = QuantPaCTActiv(abits)
        if not self.fc:
            self.mix_weight = QuantMixChanConv1d(inplane, outplane, bits=wbits, **kwargs)
        else:
            # If the layer is fc we use fixed quantization on 8bits
            self.mix_weight = QuantMixChanConv1d(inplane, outplane, 8, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        kernel_size = kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.first_layer:
            out = self.mix_activ(input)
            out = self.mix_weight(out)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8)
            out = self.mix_weight(input)
        return out


# MR
class QuantMixChanConv1d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super().__init__()
        self.bits = bits
        self.outplane = outplane

        kwargs.pop('alpha_init', None)

        self.fine_tune = kwargs.pop('fine_tune', False)
        self.conv = nn.Conv1d(inplane, outplane, **kwargs)

    def forward(self, input):
        conv = self.conv
        bias = getattr(conv, 'bias', None)
        quant_weight = _channel_asym_min_max_quantize.apply(conv.weight, self.bits)
        if bias is not None:
            quant_bias = _bias_asym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        out = F.conv1d(
            input, quant_weight, quant_bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out



# MR
class QuantMultiPrecActivConv1d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, fc=None, **kwargs):
        super().__init__()
        
        self.fine_tune = kwargs.pop('fine_tune', False)
        self.first_layer = kwargs.pop('first_layer', False)
        self.fc = fc

        self.abits = abits
        self.wbits = wbits
        
        self.search_types = ['fixed', 'mixed', 'multi']
        if fc in self.search_types:
            self.fc = fc
        else:
            self.fc = False
        self.mix_activ = QuantPaCTActiv(abits)
        if not fc:
            self.mix_weight = QuantMultiPrecConv1d(inplane, outplane, wbits, **kwargs)
        else:
            # For the final fc layer the pruning bit-width (i.e., 0) makes no sense
            _wbits = copy.deepcopy(wbits)
            if 0 in _wbits:
                _wbits.remove(0)
            # If the layer is fc we can use:
            if self.fc == 'fixed':
                # - Fixed quantization on 8bits
                self.mix_weight = QuantMixChanConv1d(inplane, outplane, 8, **kwargs)
            elif self.fc == 'mixed':
                # - Mixed-precision search 
                self.mix_weight = QuantMixChanConv1d(inplane, outplane, _wbits, **kwargs)
            elif self.fc == 'multi':
                # - Multi-precision search
                self.mix_weight = QuantMultiPrecConv1d(inplane, outplane, _wbits, **kwargs)
            else:
                raise ValueError(f"Unknown fc search, possible values are {self.search_types}")

        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        kernel_size = kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.first_layer:
            out = self.mix_activ(input)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8)
        out = self.mix_weight(out) 
        return out


# MR
class QuantMultiPrecConv1d(nn.Module):
    
    def __init__(self, inplane, outplane, bits, **kwargs):
        super().__init__()
        #assert not kwargs['bias']
        kwargs.pop('abit', None)
        if type(bits) == int:
            self.bits = [bits]
        else:
            self.bits = bits
        self.cout = outplane
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout), requires_grad=False)
        self.alpha_weight.data.fill_(0.01)
        self.conv = nn.Conv1d(inplane, outplane, **kwargs)

    def forward(self, input):
        mix_quant_weight = []
        sw = F.one_hot(torch.argmax(self.alpha_weight, dim=0), num_classes=len(self.bits)).t()
        conv = self.conv
        weight = conv.weight
        bias = getattr(conv, 'bias', None)
        for i, bit in enumerate(self.bits):
            quant_weight = _channel_asym_min_max_quantize.apply(weight, bit)
            scaled_quant_weight = quant_weight * sw[i].view((self.cout, 1, 1))
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        if bias is not None:
            quant_bias = _bias_asym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        out = F.conv1d(
            input, mix_quant_weight, quant_bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


# MR
class FpConv1d(nn.Module):

    def __init__(self, inplane, outplane, wbits, abits, first_layer=False, **kwargs):
        super().__init__()
        self.abit = abits
        self.wbits = wbits

        self.first_layer = first_layer

        self.fine_tune = kwargs.pop('fine_tune', False)
        self.fc = kwargs.pop('fc', False)

        self.conv = nn.Conv1d(inplane, outplane, **kwargs)
        self.relu = nn.ReLU()
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        kernel_size = kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.first_layer:
            out = self.conv(self.relu(input))
        else:
            out = self.conv(input)
        return out

class SharedMixQuantLinear(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantLinear, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.linear = nn.Linear(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        linear = self.linear
        weight = linear.weight
        # save repeated std computation for shared weights
        for i, bit in enumerate(self.bits):
            quant_weight =  _channel_asym_min_max_quantize.apply(weight, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.linear(input, mix_quant_weight, linear.bias)
        return out

# DJP
class QuantizedLinear(nn.Module):
    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight = True, **kwargs):
        super(QuantizedLinear, self).__init__()
        if wbits is None:
            self.wbits = 8
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = 8
        else:
            self.abits = abits
        # build mix-precision branches
        self.input_act = LearnedClippedLinearQuantization(num_bits= self.abits)
        self.linear = nn.Linear(inplane, outplane, **kwargs)
        # complexities

    def forward(self, input):
        input = self.input_act(input)
        linear = self.linear
        weight = linear.weight
        quant_weight = _channel_asym_min_max_quantize_linear.apply(weight, self.wbits)
        out = F.linear(input, quant_weight, linear.bias)
        return out

class MixActivLinear(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=True, **kwargs):
        super(MixActivLinear, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits)
        assert share_weight
        self.share_weight = share_weight
        self.mix_weight = SharedMixQuantLinear(inplane, outplane, self.wbits, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out

    def complexity_loss(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        weight_shape = list(self.mix_weight.linear.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw
