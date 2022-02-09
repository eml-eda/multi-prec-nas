import argparse
import copy
import json

import torch
import torch.nn as nn

from models.mixresnet import ResNet, BasicBlock
from models.quant_resnet import _load_arch

parser = argparse.ArgumentParser(description='Export model details for deployment')
parser.add_argument('strength', type=str, help='Strength of precision-search')

def export_onnx(model_path, precision_path):
    state_dict = torch.load(model_path)['state_dict']

    dummy_input = torch.randn(10, 3, 32, 32)
    model = ResNet(BasicBlock, nn.Conv2d, [2, 2, 2, 2], search_fc=True, num_classes=10)

    # Adapt, load pretrained state_dict and obtain clip-values
    clip_values = modify_state_dict(state_dict)

    # Load precisions
    wbits, abits = [2, 3, 4, 5, 6, 7, 8], [8]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(precision_path, name_nbits)
    archws = [wbits[w] for w in best_arch['alpha_weight']]

    quantization_detail = dict()
    i = 0
    quantized_state_dict = copy.deepcopy(state_dict)
    for name, param in clip_values.items():
        layer_name = '.'.join(name.split('.')[:-4])
        if name.split('.')[0] == 'conv1': # first layer
            layer_name = 'conv1'
        elif name.split('.')[0] == 'fc': # last layer
            layer_name = 'fc'
        quantization_detail[layer_name] = {'clip_val': [0, param], 'wbits': archws[i]}
        quantized_state_dict[layer_name+'.weight'] = quantize(state_dict[layer_name+'.weight'], archws[i])
        i += 1

    # Export quantization_detail as json
    with open('quantization_detail.json', 'w') as f:
        json.dump(quantization_detail, f, indent=4)

    # Load state_dict
    model.load_state_dict(quantized_state_dict)

    # Export onnx
    torch.onnx.export(model, dummy_input, 'resnet18_w2345678a8.onnx', verbose=True, do_constant_folding=False, training=2)

def modify_state_dict(state_dict):
    # Remove state_dict entries that are not useful, change names to match new model and save clip_values
    clip_values = {}

    state_dict_copy = copy.deepcopy(state_dict)

    for name, param in state_dict_copy.items():
        split_name = name.split('.')
        last = split_name[-1]
        if last in ['memory_size', 'size_product', 'alpha_activ']: # remove useless entries
            state_dict.pop(name)
        elif last == 'clip_val': # save clip_values
            clip_values[name] = param.cpu().item()
            state_dict.pop(name)
        elif split_name[-3:] == ['mix_weight', 'conv', 'weight']: # weight
            if split_name[0] == 'conv1': # first layer
                state_dict['conv1.weight'] = param
                # For the first layer no trainable clip_values are present, insert dummy position
                clip_values['conv1'] = 'dummy'
            elif split_name[0] == 'fc': # last layer
                state_dict['fc.weight'] = param
            else:
                state_dict['.'.join(split_name[:-3] + ['weight'])] = param
            state_dict.pop(name)
        elif split_name[-3:] == ['mix_weight', 'conv', 'bias']: # bias
            state_dict['.'.join(split_name[:-3] + ['bias'])] = param
            state_dict.pop(name)
    return clip_values

def quantize(x, bit):
    with torch.no_grad():
        ch_max, _ = x.view(x.size(0), -1).max(1)
        ch_min, _ = x.view(x.size(0), -1).min(1)

        ch_range= ch_max - ch_min
        ch_range.masked_fill_(ch_range.eq(0), 1)
        n_steps = 2 ** bit - 1
        S_w = ch_range / n_steps
        S_w = S_w.view((x.size(0), 1, 1, 1))
        y = x.div(S_w).round().mul(S_w)

        return y

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    strength = args.strength
    path = '/space/risso/multi_prec_exp/res18_w2345678a8_chan/'
    model_path = path+'model_'+strength+'/model_best.pth.tar'
    precision_path = path+'model_'+strength+'/arch_checkpoint.pth.tar'
    export_onnx(model_path, precision_path)