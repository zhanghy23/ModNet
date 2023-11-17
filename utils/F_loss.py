import torch
import torch.nn as nn
import numpy as np
class Balanceloss(nn.Module):
    def __init__(self, rou):
        super(Balanceloss,self).__init__()
        self.rou=rou
        return
    
    def forward(self,h1,h2,f1,f2):
        sigma = 0.01
        H1_r = torch.bmm(h1[:,0,:,:],f1[:,0,:,:])-torch.bmm(h1[:,1,:,:],f1[:,1,:,:])
        H1_i = torch.bmm(h1[:,0,:,:],f1[:,1,:,:])+torch.bmm(h1[:,1,:,:],f1[:,0,:,:])
        H1_h = h1[:,0,:,:]*h1[:,0,:,:]+h1[:,1,:,:]*h1[:,1,:,:]
        H1_h_diag = H1_h.diagonal(dim1=-1,dim2=-2)
        H1 = H1_r*H1_r+H1_i*H1_i
        H1_diag = H1.diagonal(dim1=-1,dim2=-2)
        loss11=torch.sum(torch.log2(1+(H1_h_diag/((torch.sum(H1_h,dim=2)-H1_h_diag)+sigma))),dim=1)
        loss12=torch.sum(torch.log2(1+(H1_diag/((torch.sum(H1,dim=2)-H1_diag)+sigma))),dim=1)
        # loss13,s=torch.min(torch.log2(1+(H1_diag/((torch.sum(H1,dim=2)-H1_diag)+sigma))),dim=1)
        #torch.sum(torch.log2(1+(H1_h_diag/((torch.sum(H1_h,dim=2)-H1_h_diag)+sigma))),dim=1)
        #torch.log2(1+torch.sum(H1_h_diag,dim=1)/(torch.sum(torch.sum(H1_h,dim=2)-H1_h_diag,dim=1)+128*sigma))
        loss1=loss11-loss12
        
        H2_r = torch.bmm(h2[:,0,:,:],f2[:,0,:,:])-torch.bmm(h2[:,1,:,:],f2[:,1,:,:])
        H2_i = torch.bmm(h2[:,0,:,:],f2[:,1,:,:])+torch.bmm(h2[:,1,:,:],f2[:,0,:,:])
        H2_h = h2[:,0,:,:]*h2[:,0,:,:]+h2[:,1,:,:]*h2[:,1,:,:]
        H2_h_diag = H2_h.diagonal(dim1=-1,dim2=-2)
        H2 = H2_r*H2_r+H2_i*H2_i
        H2_diag = H2.diagonal(dim1=-1,dim2=-2)
        t,e = torch.min(torch.log2(1+(H2_diag/((torch.sum(H2,dim=2)-H2_diag)+sigma))),dim=1)
        loss2 = torch.sum(torch.log2(1+(H2_h_diag/((torch.sum(H2_h,dim=2)-H2_h_diag)+sigma))),dim=1)-torch.sum(torch.log2(1+(H2_diag/((torch.sum(H2,dim=2)-H2_diag)+sigma))),dim=1)
        #torch.sum(torch.log2(1+(H2_h_diag/((torch.sum(H2_h,dim=2)-H2_h_diag)+sigma))),dim=1)
        MSE_r = f1[:,0,:,:]-f2[:,0,:,:]
        MSE_i = f1[:,1,:,:]-f2[:,1,:,:]
        loss3 = torch.sum(torch.sum(MSE_r*MSE_r+MSE_i*MSE_i,dim=1),dim=1)

        
        loss=torch.mean(self.rou*(loss1+loss2)+(1-self.rou)*loss3)
        loss1 = torch.mean(loss1)
        loss2 = torch.mean(loss2)
        loss3 = torch.mean(loss3)

        return loss,loss1,loss2,loss3

class Diagloss(nn.Module):
    def __init__(self):
        super(Diagloss,self).__init__()
        return
    
    def forward(self,h,x_pred):
        # sigma = 0.01
        # f = x_pred
        # # H1_r = torch.bmm(w[:,0,:,:],h[:,0,:,:])-torch.bmm(w[:,1,:,:],h[:,1,:,:])
        # # H1_i = torch.bmm(w[:,0,:,:],h[:,1,:,:])+torch.bmm(w[:,1,:,:],h[:,0,:,:])
        # # H_r = torch.bmm(H1_r,f[:,0,:,:])-torch.bmm(H1_i,f[:,1,:,:])
        # # H_i = torch.bmm(H1_i,f[:,0,:,:])+torch.bmm(H1_r,f[:,1,:,:])
        # # H = torch.complex(H_r,H_i)
        # # H_r = torch.bmm(h[:,0,:,:],f[:,0,:,:])-torch.bmm(h[:,1,:,:],f[:,1,:,:])
        # # H_i = torch.bmm(h[:,0,:,:],f[:,1,:,:])+torch.bmm(h[:,1,:,:],f[:,0,:,:])
        # I=torch.eye(128).cuda()
        # MSE_r=f[:,0,:,:]-I
        # MSE_i=f[:,1,:,:]
        # # H = torch.complex(H_r,H_i)
        # # S = torch.conj(H)
        # # H = torch.mul(H,S)
        # # H_h = h[:,0,:,:]*h[:,0,:,:]+h[:,1,:,:]*h[:,1,:,:]
        # # H_h_diag = H_h.diagonal(dim1=-1,dim2=-2)
        # # H = H_r*H_r+H_i*H_i
        # # H_diag = H.diagonal(dim1=-1,dim2=-2)
        # # loss1=torch.sum(torch.log2(1+(H_h_diag/((torch.sum(H_h,dim=2)-H_h_diag)+sigma))),dim=1)
        # # loss2=torch.sum(torch.log2(1+(H_diag/((torch.sum(H,dim=2)-H_diag)+sigma))),dim=1)
        # # loss=loss1-loss2
        # loss = torch.sum(torch.sum(MSE_r*MSE_r+MSE_i*MSE_i,dim=1),dim=1)
        # loss=torch.mean(loss)
        # # H_diag = H.diagonal(dim1=-1,dim2=-2)
        # # s=torch.diag_embed(H_diag)
        # # print(s)
        # # print(H)
        # # # H = H*torch.conj(H)
        # # loss=torch.mean(torch.square(H-s))
        sigma = 0.01
        f = x_pred
        # H1_r = torch.bmm(w[:,0,:,:],h[:,0,:,:])-torch.bmm(w[:,1,:,:],h[:,1,:,:])
        # H1_i = torch.bmm(w[:,0,:,:],h[:,1,:,:])+torch.bmm(w[:,1,:,:],h[:,0,:,:])
        # H_r = torch.bmm(H1_r,f[:,0,:,:])-torch.bmm(H1_i,f[:,1,:,:])
        # H_i = torch.bmm(H1_i,f[:,0,:,:])+torch.bmm(H1_r,f[:,1,:,:])
        # H = torch.complex(H_r,H_i)
        H_r = torch.bmm(h[:,0,:,:],f[:,0,:,:])-torch.bmm(h[:,1,:,:],f[:,1,:,:])
        H_i = torch.bmm(h[:,0,:,:],f[:,1,:,:])+torch.bmm(h[:,1,:,:],f[:,0,:,:])
        # I=torch.eye(128).cuda()
        # MSE_r=f[:,0,:,:]-I
        # MSE_i=f[:,1,:,:]
        # H = torch.complex(H_r,H_i)
        # S = torch.conj(H)
        # H = torch.mul(H,S)
        H_h = h[:,0,:,:]*h[:,0,:,:]+h[:,1,:,:]*h[:,1,:,:]
        H_h_diag = H_h.diagonal(dim1=-1,dim2=-2)
        H = H_r*H_r+H_i*H_i
        H_diag = H.diagonal(dim1=-1,dim2=-2)
        loss1=torch.sum(torch.log2(1+(H_h_diag/((torch.sum(H_h,dim=2)-H_h_diag)+sigma))),dim=1)
        loss2=torch.sum(torch.log2(1+(H_diag/((torch.sum(H,dim=2)-H_diag)+sigma))),dim=1)
        loss=loss1-loss2
        # loss = torch.sum(torch.sum(MSE_r*MSE_r+MSE_i*MSE_i,dim=1),dim=1)
        loss=torch.mean(loss)
        # H_diag = H.diagonal(dim1=-1,dim2=-2)
        # s=torch.diag_embed(H_diag)
        # print(s)
        # print(H)
        # # H = H*torch.conj(H)
        # loss=torch.mean(torch.square(H-s))
        return loss
