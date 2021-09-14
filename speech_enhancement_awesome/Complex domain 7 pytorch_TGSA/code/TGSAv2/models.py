import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from loss import SDRLoss
from hparams import hparams
import os


class Gaussian_MultiHeadSA(nn.Module): # 用于encoder
    def __init__(self,para:hparams):
        self.p = para
        super(Gaussian_MultiHeadSA, self).__init__()
        self.B = para.B # batch size
        self.D = para.D # seq dim
        self.E = para.head_nums
        #   每个SA初始化一个 惩罚 参数
        self.theta = torch.nn.Parameter(torch.tensor([para.init_gaussion_theta]), requires_grad=True)

        self.D_E = self.D // para.head_nums # head size
        self.is_punish = para.is_punish  # 惩罚机制控制

        self.query_ = nn.Linear(self.D,self.D)
        self.key_ = nn.Linear(self.D, self.D)
        self.value_ = nn.Linear(self.D, self.D)
        self.linear_o_ = nn.Linear(self.D, self.D)

        self.softmax = nn.Softmax(2) # 对每行 做softmax

    def forward(self,input,seq_len:list):

        # 通过通道数量变化来实现多头。 不需要初始化多个QKV参数。
        # 计算Q K

        this_batch_padlen = max(seq_len)
        Q = self.query_(input).permute(0,2,1).reshape(self.B*self.E,this_batch_padlen,self.D_E)#  Q  按列 切分，叠加到第一个维度
        K = self.key_(input).permute(0,2,1).reshape(self.B * self.E,this_batch_padlen,self.D_E).permute(0,2,1)
        V = self.value_(input).permute(0,2,1).reshape(self.B * self.E,this_batch_padlen,self.D_E)

        # 由于我们只需要padding mask。因此单独mask Q K就好。 最新版transformer源代码已经删除 Query mask
        ##  目的是防止 padding的数值影响到 softmax计算
        mask_q = torch.ones(self.B*self.E,this_batch_padlen,self.D_E,device=Q.device)
        for i in range(self.B):
            if seq_len[i]<this_batch_padlen:
                mask_q[i*self.E:(1+i)*self.E,seq_len[i]:,:] = 0 # 行列的seqlen:全部mask掉。

        Q = Q.mul(mask_q) # query mask 可以没有。
        K = K.mul(mask_q.transpose(2,1))  #  key mask
        att_score = torch.matmul(Q,K)/torch.sqrt(torch.tensor(this_batch_padlen,device=Q.device).float()) #QK

        if self.is_punish:
            punish_matrix = torch.exp(-((torch.arange(0,this_batch_padlen,device=Q.device).unsqueeze(0) - torch.arange(0,this_batch_padlen,device=Q.device).unsqueeze(1)) ** 2)/(self.theta** 2)).unsqueeze(0)
            punish_matrix = punish_matrix.repeat(self.B*self.E,1,1) # 重要！！！ 需要把这个矩阵扩充B*E份。
            att_score = torch.mul(att_score,punish_matrix) # GW . * QK

        # 0 的位置替换为 负无穷。
        att_score = att_score.masked_fill(att_score == 0.0, -2e20)

        att_score = self.softmax(att_score)

        A =self.linear_o_(torch.matmul(att_score,V).permute(0,2,1).reshape(self.B,self.D,this_batch_padlen).permute(0,2,1))  # re col shape
        # 经过了 attention的 input +  原来input
        return A+input,seq_len # 残差连接
        # s connection


class EncoderGSA(nn.Module):
    def __init__(self, para:hparams):

        super(EncoderGSA, self).__init__()
        self.MHSA = Gaussian_MultiHeadSA(para)

        self.linear = nn.Linear(para.D,para.D)
        # self.norm  = nn.LayerNorm()
    def forward(self,input,seq_len):
        x1,sl = self.MHSA(input,seq_len)
        x1 = nn.LayerNorm(x1.shape[-1],eps=1e-6,elementwise_affine=False)(x1) # 对最后一维度。 如果是x1.shape[1：]，则为gLN
        x2 = self.linear(x1)
        x1 = x1+x2
        x1 = nn.LayerNorm(x1.shape[-1],eps=1e-6,elementwise_affine=False)(x1)
        return x1,sl










class TGSAmodel(nn.Module):
    def __init__(self, para:hparams):
        super(TGSAmodel, self).__init__()
        self.net = []
        self.un = para.block_nums
        for i in range(self.un):
            self.net += [EncoderGSA(para)]
        self.net = nn.ModuleList(self.net)
        #  nn.ModuleList 支持多输入多输出。
        # nn.Sequential  不太支持多输出多输出。  支持单输入 单输出。

        self.lossfunc = SDRLoss()


        # init
        for p in self.parameters():
            if p.dim() > 1:
                #nn.init.uniform_(p,0,1)
                torch.nn.init.xavier_uniform(p, gain=1)
    def forward(self,input,seq_len):
        output = input
        for i in range(self.un):
            output,_ = self.net[i](output,seq_len)
        return output

    # def get_loss(self,cleanwavs:list,predwavs:list):
    #
    #     batchloss,aa = self.lossfunc(cleanwavs,predwavs)
    #     return batchloss,aa
    def get_loss(self,pred_mask:Tensor,input_mag:Tensor,input_phase:Tensor,seq_len:list,cleanwavforms,para:hparams,cleawavdirs):
        '''

        :param pred_mask:  [B,pad_T,D]
        :param input_mag: [B,pad_T,D]
        :param input_phase: [B,pad_T,D]
        :param seq_len: [B]
        :param GrifLimfunc:
        :param cleanwavforms: [B [1,T]]
        :return:
        '''
        #z=r(cosθ + isinθ)
        data_phase = torch.cos(input_phase) + torch.sin(input_phase) * 1j

        magX =torch.abs(pred_mask.mul(input_mag))  # 网络输出mask 乘以 输入的STFT mag
        spec = (magX * data_phase)  # 复数频谱
        spec = spec.permute(0,2,1) #  (B,D,T)
        predwaveforms = []
        # seq len
        # 求出Pred waveforms
        for i in range(len(seq_len)):
            speci = spec[i,:,:seq_len[i]]
            istftreuslti = torch.istft(speci,n_fft=para.n_fft)           # ISTFT输入复数谱
            predwaveforms.append(istftreuslti.unsqueeze(0))
        # padding waveforms
        clean_length_max = max([cleanwavforms[i].shape[-1] for i in range(len(cleanwavforms))])
        for i in range(len(cleanwavforms)):
            if cleanwavforms[i].shape[-1] < clean_length_max:
                cleanwavforms[i] = torch.cat([cleanwavforms[i].to(pred_mask.device),
                                              torch.zeros(1, clean_length_max - cleanwavforms[i].shape[-1],
                                                          device=pred_mask.device)], dim=1)
            else:
                cleanwavforms[i] = cleanwavforms[i][:, :clean_length_max].to(pred_mask.device)

        for i in range(len(cleanwavforms)):
            if predwaveforms[i].shape[-1] < clean_length_max:
                predwaveforms[i] = torch.cat([predwaveforms[i].to(pred_mask.device),
                                              torch.zeros(1, clean_length_max - predwaveforms[i].shape[-1],
                                                          device=pred_mask.device)], dim=1)
            else:
                predwaveforms[i] = predwaveforms[i][:, :clean_length_max].to(pred_mask.device)

        cleanwavs, predwavs = torch.cat(cleanwavforms, dim=0), torch.cat(predwaveforms, dim=0)
        cleanwavs = cleanwavs.to(pred_mask.device)
        predwavs = predwavs.to(pred_mask.device)

        batchloss = self.lossfunc(cleanwavs,predwavs) # (B ,padding_samples)

        return batchloss







# p = hparams()
# print("B is ",p.B)
# model = TGSAmodel(p)
# seq = [500,400,600,610]
# L = max(seq)
# inp = torch.rand(p.B,L,p.D)
# out = model(inp,seq)
# print(out)

# CTGSA



if __name__=="__main__":
    pass
    #check SA
    # 检测 网络的 梯度有无问题
    p = hparams(B=6)
    model = Gaussian_MultiHeadSA(p)
    # for para in model.parameters():
    #     print(para)
    # print("-----  1 网络参数打印完毕----")
    seq = [1,2,3,8,5,2]
    L = max(seq)
    inp = torch.rand(p.B,L,p.D)*100+20
    out,se = model(inp,seq)
    print("out",out)
    # loss = torch.sum((out - inp)**2)
    # print("loss is ",loss)
    # optimizer_model = optim.Adam(model.parameters(),lr = 0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0005)
    # loss.backward()
    # optimizer_model.step()
    # print("!!!!!!!!!!!!!!!step!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # for para in model.parameters():
    #     print(para)
    # print("-----  2 网络参数打印完毕----")

    # ##check Encoder
    # # 检测 encoder网络的 梯度有无问题
    # p = hparams(B=6)
    # model = EncoderGSA(p)
    # for para in model.parameters():
    #     print(para)
    # print("-----  1 网络参数打印完毕----")
    # seq = [1, 2, 3, 4, 5, 8]
    # L = max(seq)
    # inp = torch.rand(p.B, L, p.D) *2 + 1
    # out, se = model(inp, seq)
    # # print(out)
    # loss = torch.sum((out - inp) ** 2)
    # print("loss is ", loss)
    # optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
    # loss.backward()
    # optimizer_model.step()
    # print(
    #     "!!!!!!!!!!!!!!!step!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # for para in model.parameters():
    #     print(para)
    # print("-----  2 网络参数打印完毕----")

    # ##check TGSA
    # # 检测 encoder网络的 梯度有无问题
    # p = hparams(B=3,blocknums=10)
    # model = TGSAmodel(p)
    # # for para in model.parameters():
    # #     print(para)
    # # print("-----  1 网络参数打印完毕----")
    # seq = [2, 3,5]
    # L = max(seq)
    # inp = 0.01*torch.rand(p.B, L, p.D,requires_grad=True)
    # out = model(inp, seq)
    # loss2 = torch.sum((out - inp))
    # loss2.backward()
    # print("!!!!!!!!!!!!!!!back!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
    #      ' -->grad_value:',parms.grad)
    # print("loss is ", loss2)
    # print("-----  2 网络梯度打印完毕----")

    # ## check layernorm
    # B,T,D = 2,2,4
    # EPS = 1e-6
    # y = torch.randint(1,10,(B,T,D),dtype=torch.float)
    # y1,y2,y3 = y,y,y
    # print("y1  ,y1 shape",y1,y1.shape)
    # y1 = nn.LayerNorm(y1.shape[-1], eps=1e-6, elementwise_affine=False)(y1)
    # print("torch norm y1",y1)




