import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50
import numpy as np
import pandas as pd
import cv2
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from torchsummary import summary
from tensorboard.plugins import projector
from scipy.optimize import linear_sum_assignment
from utils.utils import *
from utils.assignment import *
from utils.latent_loss import *

class CNN_Feat(nn.Module):
    def __init__(self, hidden_dim=348):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, X):
        feat = self.backbone(X)
        return feat

class CNN_Text(nn.Module):
    def __init__(self, device, hidden_dim=384, nheads=4, ## According to feature vectors that we get from SBERT, hidden_dim = 384
                 num_encoder_layers=3, num_decoder_layers=3):
        super().__init__()

        # create ResNet-50 backbone
        self.conv_feat = CNN_Feat(hidden_dim)
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create encoder and decoder layers
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads)
        self.decoder = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        
        # create a default PyTorch transformer: nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder, num_decoder_layers)

        # output positional encodings (sentence)
#         self.sentence = nn.Parameter(torch.rand(20, hidden_dim))

        # spatial positional encodings (may be changed to sin positional encodings)
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
        self.device = device
        
    def forward(self, X):
        feat = self.conv_feat(X)
        feat = self.conv(feat)
        H, W = feat.shape[-2:]
        
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        feat = self.transformer_encoder(pos + 0.1 * feat.flatten(2).permute(2, 0, 1))
        R = self.transformer_decoder(torch.rand(20, feat.shape[1], feat.shape[2]).to(self.device), feat).transpose(0, 1) 
        #R = self.transformer_decoder(self.sentence.unsqueeze(1), feat).transpose(0, 1)
        return R, feat
    
class MLP(nn.Module):
    def __init__(self, hidden_dim=384):
        super().__init__()
        
        self.conv_feat = CNN_Feat(hidden_dim)
        self.linear1 = nn.Linear(131072, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, X):
        feat = self.conv_feat(X)
        x = torch.flatten(feat, 1)
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu()
        x = self.dropout(x)
        output = self.output(x).relu()
        return output