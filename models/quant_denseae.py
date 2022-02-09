import numpy as np
import torch
import torch.nn as nn
import math
from . import quant_module as qm

# MR
__all__ = [
   'quantdenseae_fp', 
]

class TinyMLDenseAe(nn.Module):
    def __init__(self, conv_func, archws, archas, qtz_fc=None, num_classes=640, input_size=640,
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
        self.gumbel = kwargs.get('gumbel', False)
        self.fc_0 = conv_func(input_size, 128, abits=archas[0], wbits=archws[0], 
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, 
                                first_layer=True, **kwargs) # 1
        self.bn_0 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_1 = conv_func(128, 128, abits=archas[1], wbits=archws[1],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 2
        self.bn_1 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_2 = conv_func(128, 128, abits=archas[2], wbits=archws[2],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 3
        self.bn_2 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_3 = conv_func(128, 128, abits=archas[3], wbits=archws[3],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 4
        self.bn_3 = nn.BatchNorm2d(128, affine=bnaff)

        self.fc_4 = conv_func(128, 8, abits=archas[4], wbits=archws[4],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 5
        self.bn_4 = nn.BatchNorm2d(8, affine=bnaff)

        self.fc_5 = conv_func(8, 128, abits=archas[5], wbits=archws[5],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 6
        self.bn_5 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_6 = conv_func(128, 128, abits=archas[6], wbits=archws[6],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 7
        self.bn_6 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_7 = conv_func(128, 128, abits=archas[7], wbits=archws[7],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 8
        self.bn_7 = nn.BatchNorm2d(128, affine=bnaff)
        self.fc_8 = conv_func(128, 128, abits=archas[8], wbits=archws[8],
                                kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=self.qtz_fc, **kwargs) # 9
        self.bn_8 = nn.BatchNorm2d(128, affine=bnaff)

        self.fc_9 = conv_func(128, num_classes, abits=archas[9], wbits=archws[9],
                                kernel_size=1, stride=1, padding=0, bias=True, groups=1, fc=self.qtz_fc, **kwargs) # 10

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
        x = self.fc_0(x)
        x = self.bn_0(x)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.fc_3(x)
        x = self.bn_3(x)
        x = self.fc_4(x)
        x = self.bn_4(x)
        x = self.fc_5(x)
        x = self.bn_5(x)
        x = self.fc_6(x)
        x = self.bn_6(x)
        x = self.fc_7(x)
        x = self.bn_7(x)
        x = self.fc_8(x)
        x = self.bn_8(x)
        x = self.fc_9(x)[:, :, 0, 0]
        return x

    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        peak_wbit = 0
        layer_idx = 0
        for layer_name, m in self.named_modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                if isinstance(m, qm.QuantMultiPrecActivConv2d):
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
                #elif isinstance(m, qm.QuantMixActivChanConv2d):
                #    import pdb; pdb.set_trace()
                #    bitops = size_product * m.abits * m.wbit
                #    bita = m.memory_size.item() * m.abits
                #    bitw = m.param_size * m.wbit
                #    sum_bitops += bitops
                #    sum_bita += bita
                #    sum_bitw += bitw
                #    peak_wbit = max(peak_wbit, bitw)
                #    if peak_wbit == bitw:
                #        peak_layer = layer_name
                else:
                    bitops = size_product * m.abits * m.wbit
                    bita = m.memory_size.item() * m.abits
                    bitw = m.param_size * m.wbit
                    #weight_shape = list(m.conv.weight.shape)
                    #print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                    #      'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, m.abit,
                    #                                   m.wbit, memory_size, m.abit, m.param_size, m.wbit))
                    sum_bitops += bitops
                    sum_bita += bita
                    sum_bitw += bitw
                    peak_wbit = max(peak_wbit, bitw)
                    if peak_wbit == bitw:
                        peak_layer = layer_name
                    layer_idx += 1
        return sum_bitops, sum_bita, sum_bitw, peak_layer, peak_wbit


def _load_arch(arch_path, names_nbits):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            assert names_nbits[name] == alpha.shape[0]
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())

    return best_arch, worst_arch

# MR
def _load_arch_multi_prec(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    best_arch['alpha_activ'], worst_arch['alpha_activ'] = [], []
    best_arch['alpha_weight'], worst_arch['alpha_weight'] = [], []
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())
        elif name == 'alpha_weight':
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax(axis=0))
            worst_arch[name].append(alpha.argmin(axis=0))

    return best_arch, worst_arch

# MR
def _load_alpha_state_dict(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ' or name == 'alpha_weight':
            alpha_state_dict[full_name] = params

    return alpha_state_dict

# MR
def quantdenseae_fp(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 10, [8] * 10
    #assert len(archas) == 10
    #assert len(archws) == 10
    return TinyMLDenseAe(qm.FpConv2d, archws, archas, **kwargs)