import torch
import torchaudio
from hparams import hparams
from torch import Tensor


'''
1、mask_score根据输入张量的各个batch 的长度mask QK矩阵。
输入 （B*E,T,T） 输出mask过的（B*E，maxT,T）。 某些值被赋值极小的数字。
        for name, param in GSAmodel.named_parameters():
            print('层:', name, param.size())
            print('权值梯度', param.grad)
        print("---------------------------------------------------------------------------------")
'''

'''
2、get_real_ima 计算波形的幅度和相位。
'''
def get_mag_phase(mono,len_frame):
    spec = torch.stft(mono, n_fft=len_frame,return_complex=True) # 返回  复数 a +b j
    mag = spec.abs()
    pha = torch.angle(spec)
    return mag,pha #论文TGSA网络要求输入。


'''
padding 将一个batch内各个T‘不同的矩阵(T’,D)padding到（T,D）
'''
def padding_input(input:list,seq_len:list):
    # 将每个batch  padding到该batch最大长度.

    L = len(seq_len)
    maxlen = max(seq_len)# 每个语音，T维，padding 到单个batch 的最大时间维度。
    output = []
    for i in range(L):
        pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, maxlen-input[i].shape[-2]))  # 用一下自带的pad函数
        output.append(pad(input[i]))
    return torch.stack(output)
  #  2  4 6 --- 6 6 6



'''
直接由音频文件路径获取到padding 的特征。
'''
def wavdir_2_padded_features(wavedirs,clean_wavdirs,para:hparams): #由多个音频路径打包成 padding矩阵。
    # 返回 mag = phase = （B*T*D）,mag用于训练， phase用于乘以网络输出
    need_pad_mag = []
    need_pad_phase = []
    sl = []
    ## 在同一函数中处理wav路劲，以确保所对比的音频是同一区间的。
    cleanwavforms = [] #
    # 获得各个语音的 张量。
    for i in range(len(wavedirs)):
        wi ,sr = torchaudio.load(wavedirs[i])
        wiclean,sr = torchaudio.load(clean_wavdirs[i])

        #长语音取其中间3秒。
        if wi.shape[-1] >= sr*para.max_time_len: # 采样率*时间 =  采样点
            indmid = wi.shape[-1]//2
            wi = wi[:,int(indmid-para.divide_radius*sr):int(indmid+para.divide_radius*sr)] # 取中间2*1.5秒。
        if wiclean.shape[-1] >= sr*para.max_time_len:
            indmid = wiclean.shape[-1] // 2
            wiclean = wiclean[:,int(indmid-para.divide_radius*sr):int(indmid+para.divide_radius*sr)] # 取中间2*1.5秒。
        wi = wi.to(para.device)

        # 语音计算特征
        cleanwavforms.append(wiclean) # 用于计算loss
        magi,phasei = get_mag_phase(wi,para.n_fft)# (1,D,T) magi\phasei 代表幅度和相位
        magi = magi.permute(0,2,1)  # (1,T,D)
        phasei = phasei.permute(0, 2, 1)  # (1,T,D)
        sl.append(magi.shape[1])  # 保存记录当前语音的T 。
        need_pad_mag.append(magi)
        need_pad_phase.append(phasei)
    # padding
    padded_mag = padding_input(need_pad_mag,sl).squeeze(1) # (B,1,T,D) -> (B,T,D)
    padded_phase = padding_input(need_pad_phase,sl).squeeze(1) # (B,1,T,D) -> (B,T,D)
    return padded_mag, padded_phase, sl,cleanwavforms


def ob_mask_distribution(input:Tensor):
    datas = []
    datas.append(input[torch.where(input < 0.0)].numel())
    datas.append(input[torch.where(input == 0.0)].numel())
    datas.append(input[torch.where((input>0.0) & (input<0.2))].numel())
    datas.append(input[torch.where((input >= 0.2) & (input < 0.4))].numel())
    datas.append(input[torch.where((input >=0.4) & (input < 0.6))].numel())
    datas.append(input[torch.where((input >= 0.6) & (input < 0.8))].numel())
    datas.append(input[torch.where((input >=0.8) & (input <= 1.0))].numel())
    datas.append(input[torch.where((input >1.0) )].numel())
    avg = torch.sum(input)/input.numel()
    return datas,avg





if __name__=="__main__":
    pass
    # # check padding
    # x1 = [torch.rand(1,200),torch.rand(1,250)]
    # x2 = [torch.rand(1,200),torch.rand(1,248)]
    # x1,x2 = padding_wavforms(x1,x2)
    # x1.requires_grad_(True)
    # x2.requires_grad_(True)
    # fenzi = torch.sum(x1**2,dim=1)
    # fenmu = torch.sum((x1-x2)**2,dim=1)
    # fenzi.retain_grad()
    # fenmu.retain_grad()
    # bs = torch.sum(10*(torch.log10(fenzi)-torch.log10(fenmu)))/x1.shape[0]
    # bs.retain_grad()
    # bs.backward()
    # # print(x1.grad, x2.grad, fenzi.grad, fenmu.grad, bs.grad)
    #
    # a =torch.randint(0,10,(2,2))
    # print(a)
    # print(torch.diag(torch.diag(a)))
    # a =   a- torch.diag(torch.diag(a))  + torch.eye(a.shape[-1])
    # print(a)