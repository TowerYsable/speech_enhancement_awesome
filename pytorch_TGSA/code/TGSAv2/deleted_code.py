
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from loss import SDRLoss
from hparams import hparams
import os

class Mask_Softmax_Input(nn.Module):
    def __init__(self, seqlen,para:hparams):
        super(Mask_Softmax_Input, self).__init__()
        self.para = para
        self.maxlen = max(seqlen)
        self.seqlen = seqlen
        self.mask = torch.zeros(para.B*para.E,self.maxlen,self.maxlen)
        self.soft = nn.Softmax(2)

        ## KEY  MASK  目的是防止 padding的数值影响到 softmax计算
        for i in range(para.B):
            if seqlen[i]<self.maxlen:
                self.mask[i*para.E:(1+i)*para.E,seqlen[i]:,:] = 1 # 行列的seqlen:全部mask掉。

    def forward(self,x):
        '''
        # x : attention score  == Q*K/sqrt(n)
        :param x: (B*E,padlen,padlen) # padlen为当前batch最大长度的语音帧数。
        :return:  (B*E,padlen,padlen)
        '''

        masked_x  = x.masked_fill(self.mask.to(self.para.device), -1e7).to(self.para.device)
        print("mask x",masked_x)
        softed_x = self.soft(torch.abs(masked_x))

        return softed_x


class Gaussian_MultiHeadSA2(nn.Module):  # 用于decoder
    def __init__(self, para: hparams):
        super(Gaussian_MultiHeadSA2, self).__init__()
        self.p = para
        self.B = para.B # batch size
        self.D = para.D # seq dim
        self.E = para.head_nums
        #   每个SA初始化一个 惩罚 参数
        self.theta = torch.nn.Parameter(torch.FloatTensor([para.init_gaussion_theta]), requires_grad=True)

        self.D_E = self.D // para.head_nums # head size
        self.is_punish = para.is_punish  # 惩罚机制控制

        self.query_ = nn.Linear(self.D,self.D)
        self.key_ = nn.Linear(self.D, self.D)
        self.value_ = nn.Linear(self.D, self.D)
        self.linear_o_ = nn.Linear(self.D, self.D)
    def forward(self,inputQ,inputK,inputV,seq_len:list):
        # 通过通道数量变化来实现多头。 不需要初始化多个QKV参数。
        # 计算Q K
        mask_soft_func = Mask_Softmax_Input(seq_len, self.p)
        this_batch_padlen = max(seq_len)
        Q = self.query_(inputQ).permute(0,2,1).reshape(self.B*self.E,this_batch_padlen,self.D_E)#  Q  按列 切分，叠加到第一个维度
        K = self.key_(inputK).permute(0,2,1).reshape(self.B * self.E,this_batch_padlen,self.D_E).permute(0,2,1)
        V = self.value_(inputV).permute(0,2,1).reshape(self.B * self.E,this_batch_padlen,self.D_E)
        x3 = torch.matmul(Q,K)/torch.sqrt(torch.tensor(this_batch_padlen,device=Q.device).float()) #QK
        if self.is_punish:
            punish_matrix = torch.exp(-((torch.arange(0,this_batch_padlen,device=Q.device).unsqueeze(0) - torch.arange(0,this_batch_padlen,device=Q.device).unsqueeze(1)) ** 2)/(self.theta** 2))
            x3 = torch.mul(x3,punish_matrix) # GW . * QK
        S = mask_soft_func(x3) # mask QK
        A =self.linear_o_(torch.matmul(S,V).permute(0,2,1).reshape(self.B,self.D,this_batch_padlen).permute(0,2,1)) # re col shape
        # 经过了 attention的 input +  原来input
        return A+inputQ,seq_len # res