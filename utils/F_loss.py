import torch
import torch.nn as nn
import numpy as np
import math
class Balanceloss(nn.Module):
    def __init__(self, rou):
        super(Balanceloss,self).__init__()
        self.rou=rou
        return
    
    def forward(self,ht1,ht2,hsc1,hsc2,wf1,wf2):
        sigma = 0.001
        H1_h = hsc1[:,0,:,:]*hsc1[:,0,:,:]+hsc1[:,1,:,:]*hsc1[:,1,:,:]
        H1_h_diag = H1_h.diagonal(dim1=-1,dim2=-2)
        #拆成w和f
        w1 = wf1[:,0:2,:,:]
        f1 = wf1[:,2:,:,:].permute(0,1,3,2)

        #分别对w和f进行功率归一化
        w1 = torch.mul(w1/torch.sqrt(torch.square(w1).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((w1.size(0),1,1,1)),math.sqrt(128))
        f1 = torch.mul(f1/torch.sqrt(torch.square(f1).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((f1.size(0),1,1,1)),math.sqrt(152))

        He11_r = torch.bmm(w1[:,0,:,:],ht1[:,0,:,:])-torch.bmm(w1[:,1,:,:],ht1[:,1,:,:])
        He11_i = torch.bmm(w1[:,0,:,:],ht1[:,1,:,:])+torch.bmm(w1[:,1,:,:],ht1[:,0,:,:])
        He1_r = torch.bmm(He11_r,f1[:,0,:,:])-torch.bmm(He11_i,f1[:,1,:,:])
        He1_i = torch.bmm(He11_r,f1[:,1,:,:])+torch.bmm(He11_i,f1[:,0,:,:])

        He1 = He1_r*He1_r+He1_i*He1_i
        He1_diag = He1.diagonal(dim1=-1,dim2=-2)
        W1 = w1[:,0,:,:]*w1[:,0,:,:]+w1[:,1,:,:]*w1[:,1,:,:]
        mm,mmm=torch.min(torch.log2(1+(He1_diag/((torch.sum(He1,dim=2)-He1_diag)+sigma*torch.sum(W1,dim=2)))),dim=1)
        # mm2,mmm=torch.min(torch.log2(1+(H1_h_diag/((torch.sum(H1_h,dim=2)-H1_h_diag)+sigma))),dim=1)
        loss1=torch.sum(torch.log2(1+(H1_h_diag/((torch.sum(H1_h,dim=2)-H1_h_diag)+sigma))),dim=1)-torch.sum(torch.log2(1+(He1_diag/((torch.sum(He1,dim=2)-He1_diag)+sigma*torch.sum(W1,dim=2)))),dim=1)+128*(-mm)

    
        H2_h = hsc2[:,0,:,:]*hsc2[:,0,:,:]+hsc2[:,1,:,:]*hsc2[:,1,:,:]
        H2_h_diag = H2_h.diagonal(dim1=-1,dim2=-2)
        #拆成w和f
        w2 = wf2[:,0:2,:,:]
        f2 = wf2[:,2:,:,:].permute(0,1,3,2)

        #分别对w和f进行功率归一化
        w2 = torch.mul(w2/torch.sqrt(torch.square(w2).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((w2.size(0),1,1,1)),math.sqrt(128))
        f2 = torch.mul(f2/torch.sqrt(torch.square(f2).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((f2.size(0),1,1,1)),math.sqrt(152))

        He12_r = torch.bmm(w2[:,0,:,:],ht2[:,0,:,:])-torch.bmm(w2[:,1,:,:],ht2[:,1,:,:])
        He12_i = torch.bmm(w2[:,0,:,:],ht2[:,1,:,:])+torch.bmm(w2[:,1,:,:],ht2[:,0,:,:])
        He2_r = torch.bmm(He12_r,f2[:,0,:,:])-torch.bmm(He12_i,f2[:,1,:,:])
        He2_i = torch.bmm(He12_r,f2[:,1,:,:])+torch.bmm(He12_i,f2[:,0,:,:])

        He2 = He2_r*He2_r+He2_i*He2_i
        He2_diag = He2.diagonal(dim1=-1,dim2=-2)
        W2 = w2[:,0,:,:]*w2[:,0,:,:]+w2[:,1,:,:]*w2[:,1,:,:]
        mm,mmm=torch.min(torch.log2(1+(He2_diag/((torch.sum(He2,dim=2)-He2_diag)+sigma*torch.sum(W2,dim=2)))),dim=1)
        # mm2,mmm=torch.min(torch.log2(1+(H2_h_diag/((torch.sum(H2_h,dim=2)-H2_h_diag)+sigma))),dim=1)
        loss2=torch.sum(torch.log2(1+(H2_h_diag/((torch.sum(H2_h,dim=2)-H2_h_diag)+sigma))),dim=1)-torch.sum(torch.log2(1+(He2_diag/((torch.sum(He2,dim=2)-He2_diag)+sigma*torch.sum(W2,dim=2)))),dim=1)+128*(-mm)


        MSE1_r = w1[:,0,:,:]-w2[:,0,:,:]
        MSE1_i = w1[:,1,:,:]-w2[:,1,:,:]
        MSE2_r = f1[:,0,:,:]-f2[:,0,:,:]
        MSE2_i = f1[:,1,:,:]-f2[:,1,:,:]
        loss3 = torch.sum(torch.sum(MSE1_r*MSE1_r+MSE1_i*MSE1_i,dim=1),dim=1)+torch.sum(torch.sum(MSE2_r*MSE2_r+MSE2_i*MSE2_i,dim=1),dim=1)

        
        loss=torch.mean(self.rou*(loss1+loss2)+(1-self.rou)*loss3)
        loss1 = torch.mean(loss1)
        loss2 = torch.mean(loss2)
        loss3 = torch.mean(loss3)

        return loss,loss1,loss2,loss3

class Diagloss(nn.Module):
    def __init__(self):
        super(Diagloss,self).__init__()
        return
    
    def forward(self,ht,hsc,wf):
        sigma = 0.01
        H_h = hsc[:,0,:,:]*hsc[:,0,:,:]+hsc[:,1,:,:]*hsc[:,1,:,:]
        H_h_diag = H_h.diagonal(dim1=-1,dim2=-2)
        loss1=torch.sum(torch.log2(1+(H_h_diag/((torch.sum(H_h,dim=2)-H_h_diag)+sigma))),dim=1)
        #拆成w和f
        w = wf[:,0:2,:,:]
        f = wf[:,2:,:,:].permute(0,1,3,2)

        #分别对w和f进行功率归一化
        w = torch.mul(w/torch.sqrt(torch.square(w).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((w.size(0),1,1,1)),math.sqrt(128))
        f = torch.mul(f/torch.sqrt(torch.square(f).sum(dim=1).sum(dim=1).sum(dim=1)).reshape((f.size(0),1,1,1)),math.sqrt(152))

        He1_r = torch.bmm(w[:,0,:,:],ht[:,0,:,:])-torch.bmm(w[:,1,:,:],ht[:,1,:,:])
        He1_i = torch.bmm(w[:,0,:,:],ht[:,1,:,:])+torch.bmm(w[:,1,:,:],ht[:,0,:,:])
        He_r = torch.bmm(He1_r,f[:,0,:,:])-torch.bmm(He1_i,f[:,1,:,:])
        He_i = torch.bmm(He1_r,f[:,1,:,:])+torch.bmm(He1_i,f[:,0,:,:])

        He = He_r*He_r+He_i*He_i
        He_diag = He.diagonal(dim1=-1,dim2=-2)
        W = w[:,0,:,:]*w[:,0,:,:]+w[:,1,:,:]*w[:,1,:,:]
        mm,mmm=torch.min(torch.log2(1+(He_diag/((torch.sum(He,dim=2)-He_diag)+sigma*torch.sum(W,dim=2)))),dim=1)
        # mm2,mmm=torch.min(torch.log2(1+(H_h_diag/((torch.sum(H_h,dim=2)-H_h_diag)+sigma))),dim=1)
        loss2=torch.sum(torch.log2(1+(He_diag/((torch.sum(He,dim=2)-He_diag)+sigma*torch.sum(W,dim=2)))),dim=1)
        loss=loss1-loss2+128*(-mm)

        loss=torch.mean(loss)

        return loss
