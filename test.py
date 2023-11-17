import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from utils.parser import args
from utils import logger
from utils.init import init_device, init_model
from dataset.dataset import datasetLoader1
from utils.statics import AverageMeter, evaluator
from models.modnet import ModNet
import math
from scipy.io import savemat

device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
PATH = r".\checkpoints\best_loss_p2.pth"
model = init_model(args)
model.to(device)
checkpoint = torch.load(PATH)
for k,v in list(checkpoint['state_dict'].items()):
        if k.startswith('Mod.'):
                s=k[len('Mod.'):]
                checkpoint['state_dict'][s]=checkpoint['state_dict'].pop(k)
torch.save(checkpoint, PATH)
print(checkpoint)
model.load_state_dict(checkpoint['state_dict'],strict=False) # 加载参数
model.eval()
model = model.to(device)
train_loader,test_loader = datasetLoader1(
        root=args.data_dir,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=pin_memory)()
model.eval()
ex_gt1=torch.zeros(1,2,128,152).to(device)
ex_gt2=torch.zeros(1,2,152,128).to(device)
for batch_idx, (ht,hsc,) in enumerate(test_loader):
        wf1 = model(ht)
        w1 = wf1[:,0:2,:,:]
        f1 = wf1[:,2:,:,:].permute(0,1,3,2)
        #分别对w和f进行功率归一化
        w1 = torch.mul(w1/torch.sqrt(torch.square(w1).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((w1.size(0),1,1,1)),math.sqrt(128))
        f1 = torch.mul(f1/torch.sqrt(torch.square(f1).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((f1.size(0),1,1,1)),math.sqrt(152))
        ex_gt1=torch.cat([ex_gt1,w1],dim=0)
        ex_gt2=torch.cat([ex_gt2,f1],dim=0)
ex_gt1=ex_gt1.cpu().detach().numpy()
ex_gt2=ex_gt2.cpu().detach().numpy()
savemat("WFtest.mat", {'W':ex_gt1,'F':ex_gt2})
