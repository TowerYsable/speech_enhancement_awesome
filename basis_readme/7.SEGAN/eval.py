import torch
import torch.nn as nn
import numpy as np
from model import Generator
from hparams import hparams
from dataset import emphasis
import glob
import soundfile as sf
import os
import librosa
import matplotlib.pyplot as plt
from numpy.linalg import norm
def enh_segan(model,noisy,para):
    # 对输入的noisy 按照 win_len 进行分段，没有重叠
   
    win_len = para.win_len
    # 不足的部分 重复填充
    N_slice = len(noisy)//win_len
    if not len(noisy)%win_len == 0:
        short = win_len - len(noisy)%win_len
        temp_noisy = np.pad(noisy,(0,short),'wrap')
        N_slice = N_slice+1
    
    slices = temp_noisy.reshape(N_slice,win_len)
  
    enh_slice = np.zeros(slices.shape)
    # # # 一次处理
    # slices = np.expand_dims(slices,axis=1)
    # slices = torch.from_numpy(slices)
    # z = nn.init.normal_(torch.Tensor(N_slice, para.size_z[0], para.size_z[1]))
    # model.eval()
    # with torch.no_grad():
        # generated_slices = model(slices, z)
    
    # generated_slices = generated_slices.numpy()
    # for n in range(N_slice):
        # generated_slice = emphasis(generated_slices[n,0,:],pre=False)
        # enh_slice[n] = generated_slice
    
    # 逐帧进行处理
    for n in range(N_slice):
        m_slice = slices[n]
      
        # 进行预加重
        m_slice = emphasis(m_slice)
        # 增加 2个维度
        m_slice = np.expand_dims(m_slice,axis=(0,1))
        # 转换为torch格式
    
        m_slice = torch.from_numpy(m_slice)
        
        # 生成 z
        z = nn.init.normal_(torch.Tensor(1, para.size_z[0], para.size_z[1]))
        
        # 进行增强
        model.eval()
        with torch.no_grad():
            generated_slice = model(m_slice, z)
        generated_slice = generated_slice.numpy()
        # 反预加重
        generated_slice = emphasis(generated_slice[0,0,:],pre=False)
        enh_slice[n] = generated_slice
    
    # 信号展开
    enh_speech = enh_slice.reshape(N_slice*win_len)
    return enh_speech[:len(noisy)]
def get_snr(clean,nosiy):
    noise = nosiy- clean
    
    snr = 20*np.log(norm(clean)/(norm(noise)+1e-7))
    return snr
    
    
if __name__ == "__main__":
    
    para = hparams()
    
    path_eval = 'eval47'
    os.makedirs(path_eval,exist_ok=True)
    
    # 加载模型
    n_epoch = 47
    model_file = "save/G_47_0.2873.pkl"
    
    generator = Generator()
    generator.load_state_dict(torch.load(model_file, map_location='cpu'))
    
    path_test_clean = '/home/sdy/dataset/enh_voicebank/clean_testset_wav/clean_testset_wav'
    path_test_noisy = '/home/sdy/dataset/enh_voicebank/noisy_testset_wav/noisy_testset_wav'
    test_clean_wavs = glob.glob(path_test_clean+'/*wav')
    # test_clean_wavs = test_clean_wavs[:15]
    fs = para.fs
    for clean_file in test_clean_wavs:
        name = os.path.split(clean_file)[-1]
        noisy_file = os.path.join(path_test_noisy,name)
        if not os.path.isfile(noisy_file):
            continue
        
        # 读取干净语音
        clean,_ = librosa.load(clean_file,sr = fs,mono=True)
        noisy,_ = librosa.load(noisy_file,sr = fs,mono=True)
        
        snr = get_snr(clean,noisy)
        print("%s  snr=%.2f"%(noisy_file,snr))
        if snr<3.0:
            print('processing %s with snr %.2f'%(noisy_file,snr))
  
            # 获取增强语音
            enh = enh_segan(generator,noisy,para)
            
            # 语音保存
            sf.write(os.path.join(path_eval,'noisy-'+name),noisy,fs)
            sf.write(os.path.join(path_eval,'clean-'+name),clean,fs)
            sf.write(os.path.join(path_eval,'enh-'+name),enh,fs)
            
            # 画频谱图
            # 绘图
            fig_name = os.path.join(path_eval,name[:-4]+'-'+str(n_epoch)+'.jpg')
            
            plt.subplot(3,1,1)
            plt.specgram(clean,NFFT=512,Fs=fs)
            plt.xlabel("clean specgram")
            plt.subplot(3,1,2)
            plt.specgram(noisy,NFFT=512,Fs=fs)
            plt.xlabel("noisy specgram")   
            plt.subplot(3,1,3)
            plt.specgram(enh,NFFT=512,Fs=fs)
            plt.xlabel("enhece specgram")
            plt.savefig(fig_name)
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    