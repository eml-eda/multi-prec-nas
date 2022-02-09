import torch
from models.mixmobilenetv1 import *

PATH = '/space/risso/multi_prec_exp/vww/mobilenetv1_w248a8_chan/model_1e-4/arch_model_best.pth.tar'

model = mixmobilenetv1_w248a8_chan(num_classes=2)
data = torch.randn(1, 3, 96, 96)

state_dict = torch.load(PATH)['state_dict']
model.load_state_dict(state_dict)

model(data)

best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.fetch_best_arch()

print(f'best_arch: {best_arch}')
print(f'bitops: {bitops}')
print(f'bita: {bita}')
print(f'bitw: {bitw}')
