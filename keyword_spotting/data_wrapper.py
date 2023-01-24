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

import numpy as np
import torch
from torch.utils.data import Dataset

class KWSDataWrapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.min = -123.5967
        self.max = 43.6677
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #data = (torch.from_numpy(self.x[idx]) - self.min) / (self.max - self.min)
        data = torch.from_numpy(self.x[idx])
        label = torch.from_numpy(np.asarray(self.y[idx]))
        return data, label

#class KWSDataWrapper(Dataset):
#    def __init__(self, data_generator):
#        self.data_generator = data_generator
#        self.min = -123.5967
#        self.max = 43.6677
#    
#    def __len__(self):
#        return len(self.data_generator)
#    
#    def __getitem__(self, idx):
#        data = torch.from_numpy(self.data_generator[idx][0])
#        #data_norm = torch.from_numpy(self.data_generator[idx][0]) - self.min
#        #data_norm = (data - self.min) / (self.max - self.min)
#        label = torch.from_numpy(self.data_generator[idx][1])
#        #return data_norm, label
#        return data, label
