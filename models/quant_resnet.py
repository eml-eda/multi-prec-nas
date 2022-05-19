import copy
import torch
import torch.nn as nn
import math
from . import quant_module as qm
from .hw_models import *

# MR
__all__ = [
    'quantres18_2w2a', 'quantres18_cfg', 'quantres18_pretrained_cfg',
    'quantres50_2w2a', 'quantres50_cfg', 'quantres50_pretrained_cfg',
    'quantres18_w248a248_multiprec', 'quantres18_w0248a248_multiprec',
    'quantres18_w8a8', 'quantres18_w4a4', 'quantres18_w2a2', 
    'quantres18_w4a8', 'quantres18_w2a8',
    'quantres18_w2345678a8_chan', 
    'quantres18_w248a8_chan', 'quantres18_w248a4_chan', 'quantres18_w248a2_chan',
    'quantres18_w248a8_chan_mp',
    'quantres18_w0248a8_multiprec', 'quantres18_w0248a4_multiprec', 'quantres18_w0248a2_multiprec',
    'quantres18_w248a8_multiprec', 'quantres18_w248a4_multiprec', 'quantres18_w248a2_multiprec',
    'quantres18_w48a8_multiprec', 'quantres18_w08a8_multiprec', 'quantres18_w024a8_multiprec',
    'quantres18_fp', 
    'quantres8_fp', 'quantres8_w2a8', 'quantres8_w4a8', 'quantres8_w8a8', 'quantres8_w32a8', 
    'quantres8_w2a4', 'quantres8_w4a4', 'quantres8_w8a4',
    'quantres8_w2a2', 'quantres8_w4a2', 'quantres8_w8a2',
    'quantres8_w0248a8_multiprec', 'quantres8_w248a8_chan', 'quantres8_w248a8_multiprec',
    'quantres8_w248a248_chan', 'quantres8_w248a248_multiprec',
]


# MR
class Backbone(nn.Module):
    def __init__(self, conv_func, input_size, bnaff, abits, wbits, **kwargs):
        super().__init__()
        self.bb_1 = BasicBlock(conv_func, 16, 16, wbits[:2], abits[:2], stride=1,
            bnaff=True, **kwargs)
        self.bb_2 = BasicBlock(conv_func, 16, 32, wbits[2:5], abits[2:5], stride=2,
            bnaff=True, **kwargs)
        self.bb_3 = BasicBlock(conv_func, 32, 64, wbits[5:7], abits[5:7], stride=2,
            bnaff=True, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=8)
    
    def forward(self, x):
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.pool(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    num_layers = 2

    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super().__init__()
        #self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv_func(inplanes, planes, archws[0], archas[0], kernel_size=3, stride=stride,
                               padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv_func(planes, planes, archws[1], archas[1], kernel_size=3, stride=1,
                               padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = conv_func(inplanes, planes, archws[-1], archas[-1], kernel_size=1,
                stride=stride, bias=False, **kwargs)
            self.bn_ds = nn.BatchNorm2d(planes)
        else:
            self.downsample = None

    def forward(self, x):
        #out = self.bn0(x)
        if self.downsample is not None:
            residual = x
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.bn_ds(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4
    num_layers = 3

    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(Bottleneck, self).__init__()
        assert len(archws) == 3
        assert len(archas) == 3
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv_func(inplanes, planes, archws[0], archas[0], kernel_size=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv_func(planes, planes, archws[1], archas[1], kernel_size=3, stride=stride,
                               padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv3 = conv_func(
            planes, planes * 4, archws[2], archas[2], kernel_size=1, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn0(x)
        if self.downsample is not None:
            residual = out
        else:
            residual = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, conv_func, layers, archws, archas, qtz_fc=None, input_size=32, num_classes=1000,
                 bnaff=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        self.inplanes = 64
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if num_classes == 1000:
            self.conv1 = conv_func(3, 64, abits=archas[0], wbits=archws[0], kernel_size=7, stride=2, padding=3, bias=False, **kwargs)
        elif num_classes == 10:
            self.conv1 = conv_func(3, 64, abits=archas[0], wbits=archws[0], kernel_size=3, stride=1, padding=3, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(64, affine=bnaff)
        self.relu = nn.ReLU(inplace=True)
        if num_classes == 1000:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        eid = block.num_layers * layers[0] + (block.expansion > 1) + 1
        self.layer1 = self._make_layer(block, conv_func, 64, layers[0], archws[1:eid], archas[1:eid],
                                       bnaff=bnaff, **kwargs)
        sid = eid
        eid = sid + block.num_layers * layers[1] + 1
        self.layer2 = self._make_layer(
            block, conv_func, 128, layers[1], archws[sid:eid], archas[sid:eid],
            stride=2, bnaff=bnaff, **kwargs
        )
        sid = eid
        eid = sid + block.num_layers * layers[2] + 1
        self.layer3 = self._make_layer(
            block, conv_func, 256, layers[2], archws[sid:eid], archas[sid:eid],
            stride=2, bnaff=bnaff, **kwargs
        )
        sid = eid
        self.layer4 = self._make_layer(block, conv_func, 512, layers[3], archws[sid:], archas[sid:],
                                       stride=2, bnaff=bnaff, **kwargs)
        #self.avgpool = nn.AvgPool2d(int(input_size/(2**5)))
        if num_classes == 1000: # ImageNet
            self.avgpool = nn.AvgPool2d(kernel_size=7)
        elif num_classes == 10: # Cifar-10
            self.avgpool = nn.AvgPool2d(kernel_size=4)
        if self.qtz_fc:
            self.fc = conv_func(512 * block.expansion, num_classes, abits=archas[-1], wbits=archws[-1], 
                kernel_size=1, stride=1, bias=False, fc=self.qtz_fc, **kwargs)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, conv_func, planes, blocks, archws, archas, stride=1,
                    bnaff=True, **kwargs):
        downsample = None
        interval = block.num_layers
        if stride != 1 or self.inplanes != planes * block.expansion:
            # the last element in arch is for downsample layer
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, archws[interval], archas[interval],
                          kernel_size=1, stride=stride, bias=False, **kwargs),
                nn.BatchNorm2d(planes * block.expansion),
            )
            archws.pop(interval)
            archas.pop(interval)

        layers = []
        #assert len(archws) == blocks * interval
        #assert len(archas) == blocks * interval
        layers.append(block(conv_func, self.inplanes, planes, archws[:interval], archas[:interval], stride,
                            downsample, bnaff=bnaff, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            sid, eid = interval * i, interval * (i + 1)
            layers.append(block(conv_func, self.inplanes, planes, archws[sid:eid], archas[sid:eid],
                                bnaff=bnaff, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.num_classes == 1000:
            x = self.maxpool(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(x)
        #assert x.shape[2] == 7
        #assert x.shape[3] == 7
        x = self.avgpool(x)
        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)
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

class TinyMLResNet(nn.Module):

    def __init__(self, block, conv_func, archws, archas, qtz_fc=None, input_size=32, num_classes=1000,
                 bnaff=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        self.inplanes = 16
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        super().__init__()
        kwargs['groups'] = 1
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv_func(3, 16, abits=archas[0], wbits=archws[0], kernel_size=3, stride=1, bias=False, padding=1, 
            first_layer=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(16, affine=bnaff)
        self.model = Backbone(conv_func, input_size, bnaff, abits=archas[1:-1], wbits=archws[1:-1], **kwargs)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #eid = block.num_layers * layers[0] + (block.expansion > 1) + 1
        #self.layer1 = self._make_layer(block, conv_func, 16, layers[0], archws[1:eid], archas[1:eid],
        #                               bnaff=bnaff, **kwargs)
        #sid = eid
        #eid = sid + block.num_layers * layers[1] + 1
        #self.layer2 = self._make_layer(
        #    block, conv_func, 32, layers[1], archws[sid:eid], archas[sid:eid],
        #    stride=2, bnaff=bnaff, **kwargs
        #)
        #sid = eid
        #eid = sid + block.num_layers * layers[2] + 1
        #self.layer3 = self._make_layer(
        #    block, conv_func, 64, layers[2], archws[sid:eid], archas[sid:eid],
        #    stride=2, bnaff=bnaff, **kwargs
        #)
        ##self.avgpool = nn.AvgPool2d(int(input_size/(2**5)))
        #self.avgpool = nn.AvgPool2d(kernel_size=8)
        if self.qtz_fc:
            self.fc = conv_func(64 * block.expansion, num_classes, abits=archas[-1], wbits=archws[-1], 
                kernel_size=1, stride=1, bias=True, fc=self.qtz_fc, **kwargs)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, conv_func, planes, blocks, archws, archas, stride=1,
                    bnaff=True, **kwargs):
        downsample = None
        interval = block.num_layers
        if stride != 1 or self.inplanes != planes * block.expansion:
            # the last element in arch is for downsample layer
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, archws[interval], archas[interval],
                          kernel_size=1, stride=stride, bias=False, **kwargs),
                nn.BatchNorm2d(planes * block.expansion),
            )
            archws.pop(interval)
            archas.pop(interval)

        layers = []
        #assert len(archws) == blocks * interval
        #assert len(archas) == blocks * interval
        layers.append(block(conv_func, self.inplanes, planes, archws[:interval], archas[:interval], stride,
                            downsample, bnaff=bnaff, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            sid, eid = interval * i, interval * (i + 1)
            layers.append(block(conv_func, self.inplanes, planes, archws[sid:eid], archas[sid:eid],
                                bnaff=bnaff, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)

        x = self.model(x)

        #x = self.relu(x)
        #assert x.shape[2] == 7
        #assert x.shape[3] == 7
        #x = self.avgpool(x)
        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)

        return x

    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_cycles = 0 
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                bitops = size_product * m.abits * m.wbit
                cycles = size_product / mpic_lut(m.abits, m.wbit)
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
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw
        #return sum_bitops, sum_bita, sum_bitw

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
def _load_weights(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    weights = {}
    for name, params in state_dict.items():
        type_ = name.split('.')[-1]
        if type_ == 'weight':
            weight = params.cpu().numpy()
            weights[name] = weight
        elif name == 'bias':
            bias = params.cpu().numpy()
            weights[name] = bias
        #elif name == 'running_mean':
        #    running_mean = params.cpu().numpy()
        #    weights[name] = running_mean
        #elif name == 'running_var':
        #    running_var = params.cpu().numpy()
        #    weights[name] = running_var

    return weights

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
# Questo era per la conv splittata in tre, per ora lo lascio ...
#def quantres18_w248a248_multiprec(arch_cfg_path, **kwargs):
#    wbits, abits = [2, 4, 8], [2, 4, 8]
#    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
#    if kwargs['fine_tune']:
#        kwargs['pretrained_weights'] = _load_weights(arch_cfg_path)
#    archas = [abits[a] for a in best_arch['alpha_activ']]
#    archws = [[(pos, wbits[w]) for pos, w in enumerate(w_ch)] for w_ch in best_arch['alpha_weight']]
#    assert len(archas) == 19
#    assert len(archws) == 19
#    return ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)

# MR
def quantres18_w248a248_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2, 4, 8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    assert len(archas) == 21
    assert len(archws) == 21
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)

    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w0248a248_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 2, 4, 8], [2, 4, 8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w48a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w08a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w024a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 2, 4], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w248a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w248a4_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [4]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w248a2_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w0248a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 2, 4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w0248a4_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 2, 4, 8], [4]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres18_w0248a2_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 2, 4, 8], [2]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 20:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 21 # 21 insead of 19 because conv1 and fc activations are also quantized
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized 
    ##

    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres8_w0248a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [0, 2, 4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 9:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 10 # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 10 # 10 instead of 8 because conv1 and fc weights are also quantized 
    ##

    model = TinyMLResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [1, 1, 1], archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
def quantres18_w248a8_chan_mp(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w in best_arch['alpha_weight']]
    #assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = ResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load all weights
        #state_dict = torch.load(arch_cfg_path)['state_dict']
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        #model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres8_w248a8_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 9:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 10 # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 10 # 10 instead of 8 because conv1 and fc weights are also quantized 
    ##

    model = TinyMLResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres8_w248a248_multiprec(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2, 4, 8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 9:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 10 # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 10 # 10 instead of 8 because conv1 and fc weights are also quantized 
    ##

    model = TinyMLResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres8_w248a8_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    #best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    #archas = [abits for a in best_arch['alpha_activ']]
    #archws = [wbits for w_ch in best_arch['alpha_weight']]
    #if len(archws) == 9:
    #    # Case of fixed-precision on last fc layer
    #    archws.append(8)
    #assert len(archas) == 10 # 10 insead of 8 because conv1 and fc activations are also quantized
    #assert len(archws) == 10 # 10 instead of 8 because conv1 and fc weights are also quantized 
    ##
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w in best_arch['alpha_weight']]

    model = TinyMLResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        checkpoint = torch.load(arch_cfg_path)
        weight_state_dict = _remove_alpha(checkpoint['state_dict'])
        model.load_state_dict(weight_state_dict, strict=False)
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        model.load_state_dict(alpha_state_dict, strict=False)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
# qtz_fc: None or 'fixed' or 'mixed' or 'multi' 
def quantres8_w248a248_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2, 4, 8]

    ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    #best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    #archas = [abits for a in best_arch['alpha_activ']]
    #archws = [wbits for w_ch in best_arch['alpha_weight']]
    #if len(archws) == 9:
    #    # Case of fixed-precision on last fc layer
    #    archws.append(8)
    #assert len(archas) == 10 # 10 insead of 8 because conv1 and fc activations are also quantized
    #assert len(archws) == 10 # 10 instead of 8 because conv1 and fc weights are also quantized 
    ##
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w in best_arch['alpha_weight']]

    model = TinyMLResNet(BasicBlock, qm.QuantMultiPrecActivConv2d, archws, archas, qtz_fc='multi', **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        checkpoint = torch.load(arch_cfg_path)
        weight_state_dict = _remove_alpha(checkpoint['state_dict'])
        model.load_state_dict(weight_state_dict, strict=False)
        alpha_state_dict = _load_alpha_state_dict_as_mp(arch_cfg_path, model)
        model.load_state_dict(alpha_state_dict, strict=False)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict = False)
    return model

# MR
def quantres18_w2a2(arch_cfg_path, **kwargs):
    archas, archws = [2] * 21, [2] * 21
    assert len(archas) == 21
    assert len(archws) == 21
    return ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)

# MR
def quantres18_w2a8(arch_cfg_path, **kwargs):
    archas, archws = [8] * 21, [2] * 21
    assert len(archas) == 21
    assert len(archws) == 21
    return ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)

# MR
def quantres18_w4a4(arch_cfg_path, **kwargs):
    archas, archws = [4] * 21, [4] * 21
    assert len(archas) == 21
    assert len(archws) == 21
    return ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)

# MR
def quantres18_w4a8(arch_cfg_path, **kwargs):
    archas, archws = [8] * 21, [4] * 21
    assert len(archas) == 21
    assert len(archws) == 21
    return ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)

# MR
def quantres18_w8a8(arch_cfg_path, **kwargs):
    archas, archws = [8] * 21, [8] * 21
    #assert len(archas) == 19
    #assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)

# MR
def quantres18_w2345678a8_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 3, 4, 5, 6, 7, 8], [8]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError()
    return model

# MR
def quantres18_w248a8_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [8]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    #assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model

# MR
def quantres18_w248a4_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    #assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError()
    return model

# MR
def quantres18_w248a2_chan(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 8], [2]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    #assert len(archas) == 21 # 21 insead of 19 because fc activations are also quantized (the first element [8] is dummy)
    #assert len(archws) == 21 # 21 instead of 19 because conv1 and fc weights are also quantized
    model = ResNet(BasicBlock, qm.QuantMixActivChanConv2d, [2, 2, 2, 2], archws, archas, qtz_fc=True, **kwargs)
    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError()
    return model

# MR
def quantres18_fp(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 21, [8] * 21
    #assert len(archas) == 19
    #assert len(archws) == 19
    return ResNet(BasicBlock, qm.FpConv2d, [2, 2, 2, 2], archws, archas, **kwargs)

# MR
def quantres8_fp(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 10, [8] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.FpConv2d, archws, archas, **kwargs)

# MR
def quantres8_w2a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 10, [2] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w4a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 10, [4] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w8a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 10, [8] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w2a4(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [4] * 10, [2] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w4a4(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [4] * 10, [4] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w8a4(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [4] * 10, [8] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w2a2(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [2] * 10, [2] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w4a2(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [2] * 10, [4] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w8a2(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [2] * 10, [8] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='mixed', **kwargs)

# MR
def quantres8_w32a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 10, [32] * 10
    assert len(archas) == 10
    assert len(archws) == 10
    return TinyMLResNet(BasicBlock, qm.QuantMixActivChanConv2d, [1, 1, 1], archws, archas, **kwargs)


def quantres18_2w2a(arch_cfg_path, **kwargs):
    archas, archws = [2] * 19, [2] * 19
    assert len(archas) == 19
    assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)


def quantres18_pretrained_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    best_activ = [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    best_weight = [0, 0, 0, 0, 2, 1, 3, 1, 0, 2, 1, 3, 1, 0, 2, 1, 3, 1, 1]
    archas = [abits[a] for a in best_activ]
    archws = [wbits[w] for w in best_weight]
    assert len(archas) == 19
    assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)


def quantres18_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 19
    assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)


def quantres50_2w2a(arch_cfg_path, **kwargs):
    archas, archws = [2] * 52, [2] * 52
    assert len(archas) == 52
    assert len(archws) == 52
    return ResNet(Bottleneck, qm.QuantActivConv2d, [3, 4, 6, 3], archws, archas, **kwargs)


def quantres50_pretrained_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    best_activ = [2, 1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1]
    best_weight = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1,
                   0, 1, 2, 2, 1, 3]
    archas = [abits[a] for a in best_activ]
    archws = [wbits[w] for w in best_weight]
    assert len(archas) == 52
    assert len(archws) == 52
    return ResNet(Bottleneck, qm.QuantActivConv2d, [3, 4, 6, 3], archws, archas, **kwargs)


def quantres50_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 52
    assert len(archws) == 52
    return ResNet(Bottleneck, qm.QuantActivConv2d, [3, 4, 6, 3], archws, archas, **kwargs)