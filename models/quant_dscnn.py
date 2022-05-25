import copy
import numpy as np
import torch
import torch.nn as nn
import math

from models.hw_models import mpic_lut
from . import quant_module as qm
from .hw_models import mpic_lut

# MR
__all__ = [
   'quantdscnn_fp', 
   'quantdscnn_w8a8', 'quantdscnn_w4a8', 'quantdscnn_w2a8',
   'quantdscnn_w8a4', 'quantdscnn_w4a4', 'quantdscnn_w2a4',
   'quantdscnn_w8a2', 'quantdscnn_w4a2', 'quantdscnn_w2a2',
   'quantdscnn_w248a8_multiprec', 'quantdscnn_w248a8_chan',
   'quantdscnn_w248a248_multiprec', 'quantdscnn_w248a248_chan',
]

# MR
class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        assert len(archws) == 2
        assert len(archas) == 2
        #self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.conv0 = conv_func(inplanes, inplanes, archws[0], archas[0], kernel_size=3, stride=stride, 
                                padding=1, bias=False, groups=inplanes, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        #self.relu0 = nn.ReLU(inplace=True)
        #self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.conv1 = conv_func(inplanes, planes, archws[1], archas[1], kernel_size=1, stride=1,
                                padding=0, bias=False, groups=1, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        #self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        #out = self.relu0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        #out = self.relu1(out)
        return out

# MR
class Backbone(nn.Module):
    def __init__(self, conv_func, input_size, bnaff, archws, archas, **kwargs):
        super().__init__()
        self.input_layer = conv_func(1, 64, abits=archas[0], wbits=archws[0],
            kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, 
            first_layer=True, **kwargs)
        self.bn = nn.BatchNorm2d(64, affine=bnaff)
        self.dpout0 = nn.Dropout(0.2)
        self.bb_1 = BasicBlock(conv_func, 64, 64, archws[1:3], archas[1:3], stride=1, **kwargs)
        self.bb_2 = BasicBlock(conv_func, 64, 64, archws[3:5], archas[3:5], stride=1, **kwargs)
        self.bb_3 = BasicBlock(conv_func, 64, 64, archws[5:7], archas[5:7], stride=1, **kwargs)
        self.bb_4 = BasicBlock(conv_func, 64, 64, archws[7:9], archas[7:9], stride=1, **kwargs)
        self.dpout1 = nn.Dropout(0.4)
        self.pool = nn.AvgPool2d((int(input_size[0]/2), int(input_size[1]/2)))
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.bn(out)
        out = self.dpout0(out)
        out = self.bb_1(out)
        out = self.bb_2(out)
        out = self.bb_3(out)
        out = self.bb_4(out)
        out = self.dpout1(out)
        out = self.pool(out)
        return out

class DS_CNN(nn.Module):
    def __init__(self, conv_func, archws, archas, qtz_fc=None, num_classes=12, input_size=(49,10),
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
        #self.model = nn.Sequential(
        #    # first_layer=True removes the ReLU activation
        #    conv_func(1, 64, abits=archas[0], wbits=archws[0], 
        #                kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, first_layer=True, **kwargs), # 1
        #    #conv_func(1, 64, abits=archas[0], wbits=archws[0], 
        #    #            kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, **kwargs), # 1
        #    nn.BatchNorm2d(64, affine=bnaff),
        #    #nn.ReLU(inplace=True),
        #    nn.Dropout(0.2),
        #    BasicBlock(conv_func, 64, 64, archws[1:3], archas[1:3], stride=1, **kwargs), # 2
        #    BasicBlock(conv_func, 64, 64, archws[3:5], archas[3:5], stride=1, **kwargs), # 3
        #    BasicBlock(conv_func, 64, 64, archws[5:7], archas[5:7], stride=1, **kwargs), # 4
        #    BasicBlock(conv_func, 64, 64, archws[7:9], archas[7:9], stride=1, **kwargs), # 5
        #    nn.Dropout(0.4), 
        #    nn.AvgPool2d((int(input_size[0]/2), int(input_size[1]/2))),
        #)
        self.model = Backbone(conv_func, input_size, bnaff, archws, archas, **kwargs)
        if self.qtz_fc:
            self.fc = conv_func(64, num_classes, abits=archas[-1], wbits=archws[-1], 
                        kernel_size=1, stride=1, padding=0, bias=True, groups=1, fc=self.qtz_fc, **kwargs)
        else:
            self.fc = nn.Linear(64, num_classes)

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
        x = self.model(x)
        #import pdb; pdb.set_trace()
        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)
        return x
    
    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_cycles = 0
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
                    if not m.first_layer:
                        cycles = size_product / mpic_lut(m.abits, m.wbit)
                    else:
                        cycles = size_product / mpic_lut(8, m.wbit)
                    bita = m.memory_size.item() * m.abits
                    bitw = m.param_size * m.wbit
                    #weight_shape = list(m.conv.weight.shape)
                    #print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                    #      'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, m.abit,
                    #                                   m.wbit, memory_size, m.abit, m.param_size, m.wbit))
                    sum_bitops += bitops
                    sum_cycles += cycles
                    sum_bita += bita
                    sum_bitw += bitw
                    peak_wbit = max(peak_wbit, bitw)
                    if peak_wbit == bitw:
                        peak_layer = layer_name
                    layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw, peak_layer, peak_wbit
        #return sum_bitops, sum_bita, sum_bitw, peak_layer, peak_wbit

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
def _load_alpha_state_dict_as_mp(arch_path, model):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha_state_dict[full_name] = params
        elif name == 'alpha_weight':
            mp_params = torch.tensor(model.state_dict()[full_name])
            mp_params[0] = params[0]
            mp_params[1] = params[1]
            mp_params[2] = params[2]
            alpha_state_dict[full_name] = mp_params

    return alpha_state_dict

# MR
def _remove_alpha(state_dict):
    weight_state_dict = copy.deepcopy(state_dict)
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            weight_state_dict.pop(full_name)
        elif name == 'alpha_weight':
            weight_state_dict.pop(full_name)

    return weight_state_dict

# MR
def quantdscnn_fp(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [8] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.FpConv2d, archws, archas, **kwargs)

# MR
def quantdscnn_w8a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [8] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w4a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [4] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w2a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [2] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w8a4(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [4] * 9, [8] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w4a4(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [4] * 9, [4] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w2a4(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [4] * 9, [2] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w8a2(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [2] * 9, [8] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w4a2(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [2] * 9, [4] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w2a2(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [2] * 9, [2] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantdscnn_w248a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    #if len(archws) == 20:
        # Case of fixed-precision on last fc layer
    #    archws.append(8)
    #assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##
    #import pdb; pdb.set_trace()
    model = DS_CNN(qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantdscnn_w248a248_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2, 4, 8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    #if len(archws) == 20:
        # Case of fixed-precision on last fc layer
    #    archws.append(8)
    #assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##
    #import pdb; pdb.set_trace()
    model = DS_CNN(qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model

# MR, as mp
def quantdscnn_w248a8_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w in best_arch['alpha_weight']]
    #assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = DS_CNN(qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        checkpoint = torch.load(arch_cfg_path)
        weight_state_dict = _remove_alpha(checkpoint['state_dict'])
        model.load_state_dict(weight_state_dict, strict=False)
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        model.load_state_dict(alpha_state_dict, strict=False)
    else:
        # Load all weights
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        #state_dict = torch.load(arch_cfg_path)['state_dict']
        #model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model

# MR, as mp
def quantdscnn_w248a248_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2, 4, 8]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w in best_arch['alpha_weight']]
    #assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = DS_CNN(qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        checkpoint = torch.load(arch_cfg_path)
        weight_state_dict = _remove_alpha(checkpoint['state_dict'])
        model.load_state_dict(weight_state_dict, strict=False)
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        model.load_state_dict(alpha_state_dict, strict=False)
    else:
        # Load all weights
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        #state_dict = torch.load(arch_cfg_path)['state_dict']
        #model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model