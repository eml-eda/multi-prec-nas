import copy
import numpy as np
import torch
import torch.nn as nn
import math
from . import quant_module_1d as qm1d

# MR
__all__ = [
   'quanttemponet_fp', 
   'quanttemponet_w2a8', 'quanttemponet_w4a8', 'quanttemponet_w8a8',
]

class TEMPONet(nn.Module):
    def __init__(self, conv_func, archws, archas, qtz_fc=None, num_classes=1, input_size=(256,),
                 bnaff=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        super().__init__()
        
        # Architecture:
        self.feature_extractor = FeatureExtractor(conv_func, input_size, archws[:9], archas[:9], **kwargs)
        self.classifier = Classifier(conv_func, self.qtz_fc, archws[9:], archas[9:], **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(1).unsqueeze(-1)
        x = self.classifier(x)
        #x = x if self.qtz_fc else x.view(x.size(0), -1)
        #x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)
        return x[:, :, 0]
    
    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        peak_wbit = 0
        layer_idx = 0
        for layer_name, m in self.named_modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                if isinstance(m, qm1d.QuantMultiPrecActivConv1d):
                    for name, params in m.state_dict().items():
                        full_name = name
                        name = name.split('.')[-1]
                        if name == 'alpha_activ':
                            abits = m.abits[torch.argmax(params, dim=0).item()]
                        elif name == 'alpha_weight':
                            wbits = [m.wbits[prec] for prec in torch.argmax(params, dim=0).tolist()]
                    wbit_eff = sum(wbits) / len(wbits)
                    bitops = size_product * abits * wbit_eff
                    bita = m.memory_size.item() * abits
                    bitw = m.param_size * wbit_eff
                    sum_bitops += bitops
                    sum_bita += bita
                    sum_bitw += bitw
                    peak_wbit = max(peak_wbit, bitw)
                    if peak_wbit == bitw:
                        peak_layer = layer_name
                else:
                    bitops = size_product * m.abits * m.wbit
                    bita = m.memory_size.item() * m.abits
                    bitw = m.param_size * m.wbit
                    sum_bitops += bitops
                    sum_bita += bita
                    sum_bitw += bitw
                    peak_wbit = max(peak_wbit, bitw)
                    if peak_wbit == bitw:
                        peak_layer = layer_name
                    layer_idx += 1
        return sum_bitops, sum_bita, sum_bitw, peak_layer, peak_wbit

class FeatureExtractor(nn.Module):

    def __init__(self, conv_func, input_size, archws, archas, **kwargs):
        super().__init__()
        # pad = ((k-1)*dil+1)//2
        # First Stack
        self.tcb00 = TempConvBlock(conv_func, ch_in=4, ch_out=32, k=3, 
            dil=2, pad=2, stride=1, archws=archws[0], archas=archas[0], first_layer=True, **kwargs)
        self.tcb01 = TempConvBlock(conv_func, ch_in=32, ch_out=32, k=3, 
            dil=2, pad=2, stride=1, archws=archws[1], archas=archas[1], **kwargs)
        self.cb0 = ConvBlock(conv_func, ch_in=32, ch_out=64, k=5,
            dil=1, pad=2, stride=1, archws=archws[2], archas=archas[2], **kwargs)
        # Second Stack
        self.tcb10 = TempConvBlock(conv_func, ch_in=64, ch_out=64, k=3, 
            dil=4, pad=4, stride=1, archws=archws[3], archas=archas[3], **kwargs)
        self.tcb11 = TempConvBlock(conv_func, ch_in=64, ch_out=64, k=3, 
            dil=4, pad=4, stride=1, archws=archws[4], archas=archas[4], **kwargs)
        self.cb1 = ConvBlock(conv_func, ch_in=64, ch_out=128, k=5,
            dil=1, pad=2, stride=2, archws=archws[5], archas=archas[5], **kwargs)
        # Third Stack
        self.tcb20 = TempConvBlock(conv_func, ch_in=128, ch_out=128, k=3, 
            dil=8, pad=8, stride=1, archws=archws[6], archas=archas[6], **kwargs)
        self.tcb21 = TempConvBlock(conv_func, ch_in=128, ch_out=128, k=3, 
            dil=8, pad=8, stride=1, archws=archws[7], archas=archas[7], **kwargs)
        self.cb2 = ConvBlock(conv_func, ch_in=128, ch_out=128, k=5,
            dil=1, pad=4, stride=4, archws=archws[8], archas=archas[8], **kwargs)

    def forward(self, x):
        x = self.tcb00(x)
        x = self.tcb01(x)
        x = self.cb0(x)
        x = self.tcb10(x)
        x = self.tcb11(x)
        x = self.cb1(x)
        x = self.tcb20(x)
        x = self.tcb21(x)
        x = self.cb2(x)
        return x


class TempConvBlock(nn.Module):
    def __init__(self, conv_func, ch_in, ch_out, k, dil, pad, stride, archws, archas, first_layer=False, **kwargs):
        super().__init__()
        self.tcn = conv_func(ch_in, ch_out, archws, archas, kernel_size=k, 
            dilation=dil, stride=stride, padding=pad, bias=False, first_layer=first_layer, **kwargs)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = self.tcn(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, conv_func, ch_in, ch_out, k, dil, pad, stride, archws, archas, **kwargs):
        super().__init__()
        self.conv = conv_func(ch_in, ch_out, archws, archas, kernel_size=k, 
            dilation=dil, stride=stride, padding=pad, bias=False, **kwargs)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, conv_func, qtz_fc, archws, archas, **kwargs):
        super().__init__()
        self.fc0 = conv_func(512, 256, abits=archas[0], wbits=archws[0], 
            kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=qtz_fc, **kwargs)
        self.bn0 = nn.BatchNorm1d(256)
        self.fc1 = conv_func(256, 128, abits=archas[1], wbits=archws[1],
            kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=qtz_fc, **kwargs)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = conv_func(128, 1, abits=archas[2], wbits=archws[2],
            kernel_size=1, stride=1, padding=0, bias=True, groups=1, fc=qtz_fc, **kwargs)
    
    def forward(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        return x


def quanttemponet_fp(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 12, [8] * 12
    return TEMPONet(qm1d.FpConv1d, archws, archas, **kwargs)

# MR
def quanttemponet_w2a8(arch_cfg_path, **kwargs):
    archas, archws = [8] * 12, [2] * 12
    return TEMPONet(qm1d.QuantMixActivChanConv1d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quanttemponet_w4a8(arch_cfg_path, **kwargs):
    archas, archws = [8] * 12, [4] * 12
    return TEMPONet(qm1d.QuantMixActivChanConv1d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quanttemponet_w8a8(arch_cfg_path, **kwargs):
    archas, archws = [8] * 12, [8] * 12
    return TEMPONet(qm1d.QuantMixActivChanConv1d, archws, archas, qtz_fc='mixed', **kwargs)