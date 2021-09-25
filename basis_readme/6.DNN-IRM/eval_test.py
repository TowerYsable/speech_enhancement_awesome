import torch
from hparams import hparams
from dataset import feature_stft, feature_contex,get_mask
import os
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from generate_training import signal_by_db


def eval_file_IRM(wav_file,model,para):
    
    # 读取noisy 的音频文件
    noisy_wav,fs = sf.read(wav_file,dtype = 'int16')
    noisy_wav = noisy_wav.astype('float32')

    # 提取LPS特征
    noisy_mag,noisy_phase = feature_stft(noisy_wav,para.para_stft)

    # 转为torch格式
    noisy_LPS = torch.from_numpy(np.log(noisy_mag**2))

    # 进行拼帧
    noisy_LPS_expand = feature_contex(noisy_LPS,para.n_expand)
    
    # 利用DNN进行mask计算
    model.eval()
    with torch.no_grad():
        enh_mask = model(x = noisy_LPS_expand)
    # 转为numpy格式
    enh_mask = enh_mask.numpy()

    enh_pahse = noisy_phase[para.n_expand:-para.n_expand,:].T
    enh_mag = (noisy_mag[para.n_expand:-para.n_expand,:]*enh_mask).T

    enh_spec = enh_mag*np.exp(1j*enh_pahse)

    # istft
    enh_wav = librosa.istft(enh_spec, hop_length=para.para_stft["hop_length"], win_length=para.para_stft["win_length"])
    return enh_wav
    
def eval_file_IRM_test(wav_file,clean_file,para):
    
    # 读取noisy 的音频文件
    noisy_wav,fs = sf.read(wav_file,dtype = 'float32')
    noisy_wav = noisy_wav.astype('float32')

    clean_wav,fs = sf.read(clean_file,dtype = 'float32')
    clean_wav = clean_wav.astype('float32')
    
    _,noisy_phase = feature_stft(noisy_wav,para.para_stft)
    
    clean_mag,noisy_mag,mask = get_mask(clean_wav,noisy_wav,para.para_stft)
    
  
    enh_pahse = noisy_phase.T
    enh_mag = (noisy_mag*mask**0.5).T

    enh_spec = enh_mag*np.exp(1j*enh_pahse)

    # istft
    enh_wav = librosa.istft(enh_spec, hop_length=para.para_stft["hop_length"], win_length=para.para_stft["win_length"])
    
    return enh_wav    
    
   
    
if __name__ == "__main__":
    
    para = hparams()
    
    # 读取训练好的模型
    model_name = "save/model_3_0.0926.pth"
    m_model = torch.load(model_name,map_location = torch.device('cpu'))
    
    snrs = [0,5]
    noise_path = '/home/sdh/dataset/noise/'
    noises = ['white']
    
    test_clean_files = np.loadtxt('scp/test_small.scp',dtype = 'str').tolist()
    test_clean_files = test_clean_files[:2]
    path_eval = 'eval_test'
    os.makedirs(path_eval,exist_ok=True)
    clean_path = '/home/sdh/dataset/TIMIT'
    
    for noise in noises:
        print(noise)
        noise_file = os.path.join(noise_path,noise+'.wav')
        noise_data,fs = sf.read(noise_file,dtype = 'int16')
        
        for clean_wav in test_clean_files:
            
            # 读取干净语音并保存
            clean_file = os.path.join(clean_path,clean_wav)
            clean_data,fs = sf.read(clean_file,dtype = 'int16')
            id = os.path.split(clean_file)[-1]
            sf.write(os.path.join(path_eval,id),clean_data,fs)

            for snr in snrs:
                # 生成noisy文件
                noisy_file = os.path.join(path_eval,noise+'-'+str(snr)+'-'+id)
                mix = signal_by_db(clean_data,noise_data,snr)
                noisy_data = np.asarray(mix,dtype= np.int16)
                sf.write(noisy_file,noisy_data,fs)
                
                # 进行增强
                print("enhancement file %s"%(noisy_file))
                enh_data = eval_file_IRM_test(noisy_file,clean_file,para)
                
                # 信号正则
                # max_ = np.max(enh_data)
                # min_ = np.min(enh_data)
                # enh_data = enh_data*(2/(max_ - min_)) - (max_+min_)/(max_-min_)
                enh_file = os.path.join(path_eval,noise+'-'+str(snr)+'-'+'enh'+'-'+id)
                sf.write(enh_file,enh_data,fs)
                
                # 绘图
                fig_name = os.path.join(path_eval,noise+'-'+str(snr)+'-'+id[:-3]+'jpg')
                
                plt.subplot(3,1,1)
                plt.specgram(clean_data,NFFT=512,Fs=fs)
                plt.xlabel("clean specgram")
                plt.subplot(3,1,2)
                plt.specgram(noisy_data,NFFT=512,Fs=fs)
                plt.xlabel("noisy specgram")   
                plt.subplot(3,1,3)
                plt.specgram(enh_data,NFFT=512,Fs=fs)
                plt.xlabel("enhece specgram")
                plt.savefig(fig_name)
                
                
                
                
               
    
 
    
    