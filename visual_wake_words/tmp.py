import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch

from models.mixmobilenetv1 import mixmobilenetv1_w0248a8_multiprec

model = mixmobilenetv1_w0248a8_multiprec(num_classes=2, 
                                        reg_target='weights',
                                        alpha_init='same', 
                                        gumbel=False)

path = '/space/risso/multi_prec_exp/vww/mobilenetv1_w0248a8_multiprec/warmup_8bit.pth.tar'
checkpoint = torch.load(path)['state_dict']

import pdb; pdb.set_trace()

model.load_state_dict(checkpoint, strict=False)
    
pdb.set_trace()