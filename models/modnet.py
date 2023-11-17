r""" The proposed CRNet
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import math

from utils import logger

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class ModNet(nn.Module):
    def __init__(self):
        super(ModNet, self).__init__()

        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv", ConvBN(2,4,7)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3)),      
        ]))

        self.encoder2 = nn.Sequential(OrderedDict([
            ("conv", ConvBN(6,2,7)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3)),      
        ]))

        self.encoder3 = nn.Sequential(OrderedDict([
            ("conv", ConvBN(8,2,7)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3)),      
        ]))

        self.encoder_fc1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(2*152*152, 4*152)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3)),      
        ]))         
        self.encoder_fc2 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(4*152, 4*152)),     
            ("relu1", nn.LeakyReLU(negative_slope=0.3)), 
        ]))   
        
        self.encoder_fc = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(4*152,4*128*152)),
            # ("BN", nn.BatchNorm1d(2*128*128)),      
        ])) 
        
            
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, h):
        n, c, a, b = h.detach().size()
        out1 = self.encoder1(h)
        out2 = self.encoder2(torch.cat([out1,h],dim=1))
        out3 = self.encoder3(torch.cat([h,out1,out2],dim=1))
        out = self.encoder_fc1(out3.view(n,-1))
        out = self.encoder_fc2(out)
        out = self.encoder_fc(out).view(n,4,128,152)
        # out = torch.mul(out/torch.sqrt(torch.square(out).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((n,1,1,1)),math.sqrt(128))
        return out

class ModNets(nn.Module):
    def __init__(self):
        super(ModNets, self).__init__()
        self.Mod = modnet(phase=2)

    def forward(self, h1, h2):
        n, c, a, b = h1.detach().size()
        outh1 = self.Mod.encoder1(h1)
        outh2 = self.Mod.encoder2(torch.cat([outh1,h1],dim=1))
        outh3 = self.Mod.encoder3(torch.cat([h1,outh1,outh2],dim=1))
        outh = self.Mod.encoder_fc1(outh3.view(n,-1))
        outh = self.Mod.encoder_fc2(outh)
        outh = self.Mod.encoder_fc(outh).view(n,4,128,152)
        # outh = torch.mul(outh/torch.sqrt(torch.square(outh).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((n,1,1,1)),math.sqrt(128))

        out1 = self.Mod.encoder1(h2)
        out2 = self.Mod.encoder2(torch.cat([out1,h2],dim=1))
        out3 = self.Mod.encoder3(torch.cat([h2,out1,out2],dim=1))
        out = self.Mod.encoder_fc1(out3.view(n,-1))
        out = self.Mod.encoder_fc2(out)
        out = self.Mod.encoder_fc(out).view(n,4,128,152)
        # out = torch.mul(out/torch.sqrt(torch.square(out).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((n,1,1,1)),math.sqrt(128))

        return outh,out

def modnet(phase):
    r""" Create a proposed CRNet.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of CRNet
    """
    model = ModNet()
    if phase==2:
        PATH = r".\checkpoints\best_loss_p1.pth"
        checkpoint = torch.load(PATH)
        for k,v in list(checkpoint['state_dict'].items()):
            if k.startswith('Mod.'):
                s=k[len('Mod.'):]
                checkpoint['state_dict'][s]=checkpoint['state_dict'].pop(k)
        model.load_state_dict(checkpoint['state_dict'],strict=False) # 加载参数
    return model

def modnets():
    model = ModNets()
    return model
