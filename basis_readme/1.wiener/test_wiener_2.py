import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def sub_spec(noisy,noise,para):
    n_fft = para["n_fft"]
    hop_length = para["hop_length"]
    win_length = para["win_length"]
    
    # 计算noisy的频谱
    S_noisy = librosa.stft(noisy,n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # D x T
    D,T = np.shape(S_noisy)
    Mag_noisy= np.abs(S_noisy)
    Phase_nosiy= np.angle(S_noisy)
    Power_nosiy = Mag_noisy**2
    
    # 计算noise的频谱
    S_noise = librosa.stft(noise,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    Mag_nosie = np.mean(np.abs(S_noise),axis=1,keepdims=True)
    Power_nosie = Mag_nosie**2
    Power_nosie = np.tile(Power_nosie,[1,T])
    
    ## 方法3 引入平滑
    Mag_noisy_new = np.copy(Mag_noisy)
    k = para["k"]
    for t in range(k,T-k):
        Mag_noisy_new[:,t] = np.mean(Mag_noisy[:,t-k:t+k+1],axis=1)
    Power_nosiy = Mag_noisy_new**2
    
    # 超减法去噪
    alpha = para["alpha"]
    gamma = para["gamma"]

    Power_enhenc = np.power(Power_nosiy,gamma) - alpha*np.power(Power_nosie,gamma)
    Power_enhenc = np.power(Power_enhenc,1/gamma)
    
    # 对于过小的值用 beta* Power_nosie 替代
    beta = para["beta"]
    mask = (Power_enhenc>=beta*Power_nosie)-0
    Power_enhenc = mask*Power_enhenc + beta*(1-mask)*Power_nosie
    Mag_enhenc = np.sqrt(Power_enhenc)
    
    
    Mag_enhenc_new  = np.copy(Mag_enhenc)
    # 计算最大噪声残差
    maxnr = np.max(np.abs(S_noisy[:,:31])-Mag_nosie,axis =1)

    k = 1
    for t in range(k,T-k):
        index = np.where(Mag_enhenc[:,t]<maxnr)[0]
        temp = np.min(Mag_enhenc[:,t-k:t+k+1],axis=1)
        Mag_enhenc_new[index,t] = temp[index]
            
    # 对信号进行恢复
    S_enhec = Mag_enhenc_new*np.exp(1j*Phase_nosiy)
    enhenc = librosa.istft(S_enhec, hop_length=128, win_length=256)
    
    return enhenc
  
def wiener_filter(noisy,clean,noise,para):
    n_fft = para["n_fft"]
    hop_length = para["hop_length"]
    win_length = para["win_length"]
    alpha = para["alpha"]
    beta = para["beta"]
    
    S_noisy = librosa.stft(noisy,n_fft=n_fft, hop_length=hop_length, win_length=win_length)  #DxT
    S_noise = librosa.stft(noise,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_clean = librosa.stft(clean,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    Pxx = np.mean((np.abs(S_clean))**2,axis=1,keepdims=True) # Dx1
    Pnn = np.mean((np.abs(S_noise))**2,axis=1,keepdims=True)
   
    H = (Pxx/(Pxx+alpha*Pnn))**beta
    
    S_enhec = S_noisy*H
    
    enhenc = librosa.istft(S_enhec, hop_length=hop_length, win_length=win_length)
    
    return H,enhenc
    
if __name__ == "__main__":
    
    # 读取干净语音
    clean_wav_file = "sp01.wav"
    clean,fs = librosa.load(clean_wav_file,sr=None) 
    
    # 读取读取噪声语音
    noisy_wav_file = "in_SNR5_sp01.wav"
    noisy,fs = librosa.load(noisy_wav_file,sr=None)
    
    
    # 设置谱减法模型参数
    para_sub_spec = {}
    para_sub_spec["n_fft"] = 256
    para_sub_spec["hop_length"] = 128
    para_sub_spec["win_length"] = 256
    para_sub_spec["alpha"] = 4
    para_sub_spec["beta"] = 0.0001
    para_sub_spec["gamma"] =1 
    para_sub_spec["k"] =1
    
    # 利用谱减法估计噪声
    # 前5000点 大约30帧作为噪声
    est_clean = sub_spec(noisy,noisy[:2000],para_sub_spec)
    
    est_noise = noisy[:len(est_clean)]-est_clean
    
    # 设置维纳滤波模型参数
    para_wiener = {}
    para_wiener["n_fft"] = 256
    para_wiener["hop_length"] = 128
    para_wiener["win_length"] = 256
    para_wiener["alpha"] = 1
    para_wiener["beta"] =1
    
    # 维纳滤波
    H,enhenc = wiener_filter(noisy,est_clean,est_noise,para_wiener)
    
    sf.write("enhce_2.wav",enhenc,fs)

    
    plt.subplot(3,1,1)
    plt.specgram(clean,NFFT=256,Fs=fs)
    plt.xlabel("clean specgram")
    plt.subplot(3,1,2)
    plt.specgram(noisy,NFFT=256,Fs=fs)
    plt.xlabel("noisy specgram")   
    plt.subplot(3,1,3)
    plt.specgram(enhenc,NFFT=256,Fs=fs)
    plt.xlabel("enhece specgram")  
    plt.show()
    
    
    