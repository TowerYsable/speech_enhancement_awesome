import torch.optim as optim
import torch
import numpy
import torch.nn as nn
from torch import Tensor



class SDRLoss(nn.Module): #计算一batch语音的loss
    def __init__(self ):
        super(SDRLoss, self).__init__()
    def forward(self, cleanwavforms:Tensor,pred_waveforms:Tensor):  # 输入的是 clean和pred的padding
        # 输入2个 [B,T]
        batchsize = cleanwavforms.shape[0]
        fenzi = torch.sum( ((torch.sum(torch.mul(cleanwavforms,pred_waveforms),dim=1) / torch.sum(cleanwavforms**2,dim=1)  ).unsqueeze(1).mul(cleanwavforms))**2,dim=1)
        fenmu =  torch.sum( (  (torch.sum(torch.mul(cleanwavforms,pred_waveforms),dim=1) / torch.sum(cleanwavforms**2,dim=1)  ).unsqueeze(1).mul(cleanwavforms) - pred_waveforms )**2,dim=1)


        bs = ( torch.sum(10 * (torch.log10(fenzi) - torch.log10(fenmu))  ) )/batchsize

        return -bs # !由于 我们的指标SDR是随着模型效果的增强而增大，因此我们设 loss = - SDR


if __name__=="__main__":
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(10)
    n = 16000
    B = 2
    a = torch.randint(1,5,(B,n))
    b = a + torch.rand(B,n)
    print("a  :",a)
    print("b  :",b)
    print("---")
    fenzi = torch.sum( ((torch.sum(torch.mul(a,b),dim=1) / torch.sum(a**2,dim=1)  ).unsqueeze(1).mul(a))**2,dim=1)
    print(fenzi)
    fenmu =  torch.sum( (  (torch.sum(torch.mul(a,b),dim=1) / torch.sum(a**2,dim=1)  ).unsqueeze(1).mul(a) - b )**2,dim=1)
    print(fenmu)
    print(fenzi/(fenmu+1e-7))
    print(( torch.sum(10 * (torch.log10(fenzi) - torch.log10(fenmu))) )/B)

    print("sdrloss func")
    sdrloss = SDRLoss()
    print(sdrloss(a,b))

