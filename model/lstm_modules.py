import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from .utils import *
from .resnet_ibn import *

# 2D CNN encoder using ResNet-50 pretrained
class IBN_Encoder(nn.Module):
    def __init__(self, drop_p=0.25, CNN_embed_dim=512, mode='ibnnet'):
        super(IBN_Encoder, self).__init__()

        self.drop_p = drop_p
        if mode=='ibnnet':
            resnet = resnet50_ibn_a(pretrained=True)
        else:
            resnet = resnet = models.resnet50(pretrained=True)
        resnet.avgpool = AdaptiveConcatPool2d()
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.headbn1 = nn.BatchNorm1d(4096)
        self.headdr1 = nn.Dropout(p=self.drop_p)
        self.fc1 = nn.Linear(4096, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.headbn1(x)
            x = self.fc1(x)

            cnn_embed_seq.append(x)
            #print(type(ccn_embed_seq))

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        return cnn_embed_seq



class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=512, h_RNN_layers=3, h_RNN=512, drop_p=0.2, num_classes=5):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=self.h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        return x
    
    
    