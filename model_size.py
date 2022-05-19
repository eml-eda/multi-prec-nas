import argparse
import torch

import models as models

parser = argparse.ArgumentParser(description='Get model size')
parser.add_argument('arch', type=str, help='Architecture name')
parser.add_argument('--num-classes', type=int, help='Number of output classes')
parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model path')

args = parser.parse_args()
print(args)

# Build model
#model = models.__dict__[args.arch]('', num_classes=10)

# Load pretrained model if specified
if args.pretrained_model is not None:
    model = models.__dict__[args.arch](args.pretrained_model, num_classes=args.num_classes, fine_tune=False)
else:
    model = models.__dict__[args.arch]('', num_classes=args.num_classes)

# Feed random input
model_name = str(args.arch).split('quant')[1].split('_')[0]
if model_name == 'mobilenetv1':
    rnd_input = torch.randn(1, 3, 96, 96)
elif model_name == 'resnet18' or model_name == 'res8':
    rnd_input = torch.randn(1, 3, 32, 32)
elif model_name == 'dscnn':
    rnd_input = torch.randn(1, 1, 49, 10)
elif model_name == 'denseae':
    rnd_input = torch.randn(2, 640, 1, 1)
elif model_name == 'temponet':
    rnd_input = torch.randn(2, 4, 256)
else:
    raise ValueError(f'Unknown model name: {model_name}')
with torch.no_grad():
    model(rnd_input)

try:
    cycles, bita, bitw, peak_mem_layer, peak_mem_bitw = model.fetch_arch_info()
except:
    cycles, bita, bitw = model.fetch_arch_info()

print(f'cycles: {cycles}')
print(f'bita: {bita}')
print(f'bitw: {bitw}')
print(f'peak_mem_bitw: {peak_mem_bitw} @ {peak_mem_layer}')