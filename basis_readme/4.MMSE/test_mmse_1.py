import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.integrate as inte
def Gan_mmse(ksi,gamma):
    c = np.sqrt(np.pi) / 2
    
    v = gamma * ksi / (1 + ksi)
    
    j_0 = sp.iv(0 , v/2)  
    j_1 = sp.iv(1 , v/2)    
    C = np.exp(-0.5 * v)
    A = ((c * (v ** 0.5)) * C) / gamma      #[7.40] A
    B = (1 + v) * j_0 + v * j_1             #[7.40] B
    hw = A * B    #[7.40]
    return hw
    
def Gan_log_mmse(ksi,gamma):
    def integrand(t):
        return np.exp(-t) / t
    A = ksi / (1 + ksi) 
    v = A * gamma
    ei_v = np.zeros(len(v))
    for i in range(len(v)): 
        ei_v[i] = 0.5 * inte.quad(integrand,v[i],np.inf)[0]
    hw = A * np.exp(ei_v)
    return hw
    
def Gan_log_mmse2(ksi,gamma):
  
    A = ksi / (1 + ksi) 
    v = A * gamma
    ei_v = np.zeros(len(v))
    for i in range(len(v)): 
        
        if v[i]<0.1:
            ei_v[i] = -2.3*np.log10(v[i])-0.6
        elif v[i]>=0.1 and v[i]<1:
            ei_v[i] = -1.544*np.log10(v[i]) + 0.166
        else:
            ei_v[i] = np.power(10,-0.53*v[i]-0.26)
      
    hw = A * np.exp(0.5*ei_v)
    return hw

def Gan_sqr_mmse(ksi,gamma):
   
    A = ksi / (1 + ksi) 
    v = A * gamma
    
    B = (1 + v) / gamma
    hw = np.sqrt(A * B)
    
    
    return hw
    

def enh_mmse(noisy,noise,para):
    n_fft = para["n_fft"]
    hop_length = para["hop_length"]
    win_length = para["win_length"]
    
    S_noisy = librosa.stft(noisy,n_fft=n_fft, hop_length=hop_length, win_length=win_length)  #DxT
    S_noise = librosa.stft(noise,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    phase_nosiy = np.angle(S_noisy)
    mag_noisy = np.abs(S_noisy)
    
    D,T = np.shape(mag_noisy)
    
    
    mag_nosie = np.mean(np.abs(S_noise),axis=1)
    power_noise = mag_nosie**2
    
    mag_enhance = np.zeros([D,T])
    aa = para["a_DD"]
    
    for i in range(T):
        
        # 获取每一帧的 能量谱和幅度谱
        mag_frame = mag_noisy[:,i]
        power_frame = mag_frame**2
        
        # 获取用来进行 VAD 计算的 信噪比
        SNR_VAD = 10 * np.log10(np.sum(power_frame)/np.sum(power_noise))
        
        # 计算后验信噪比
        gamma = np.minimum(power_frame / power_noise , para["max_gamma"])
        
        # 计算先验信噪比
        if i == 0:
            ksi = aa + (1 - aa) * np.maximum(gamma - 1 , 0)
        else:
            ksi = aa * power_enhance_frame / power_noise + (1 - aa) * np.maximum(gamma - 1 , 0)
            # 对 ksi 的最小值进行限制 
            ksi = np.maximum(para["ksi_min"] , ksi)

        # 根据 VAD 更新 power_noise
        mu = para["mu_VAD"]
        if SNR_VAD < para["th_VAD"]:  
            power_noise = mu * power_noise + (1 - mu) * power_frame  
        
        H = para["fun_GAN"] (ksi,gamma)
        
        mag_enhance_frame = H * mag_frame
        mag_enhance[:,i] = mag_enhance_frame
        
        power_enhance_frame = mag_enhance_frame ** 2
    
    S_enhec = mag_enhance*np.exp(1j*phase_nosiy)
    
    enhance = librosa.istft(S_enhec, hop_length=hop_length, win_length=win_length)
    return enhance
    
    
    
if __name__ == "__main__":
    
    # 读取干净语音
    clean_wav_file = "sp01.wav"
    clean,fs = librosa.load(clean_wav_file,sr=None) 
    print(fs)
    # 读取读取噪声语音
    noisy_wav_file = "in_SNR5_sp01.wav"
    noisy,fs = librosa.load(noisy_wav_file,sr=None)
    
    
    # 设置模型参数
    para_mmse = {}
    para_mmse["n_fft"] = 256
    para_mmse["hop_length"] = 128
    para_mmse["win_length"] = 256
    para_mmse["max_gamma"] =40 # gamma 的最大值
    para_mmse["a_DD"] = 0.98  # 利用 decision-direct 进行 ksi更新的参数
    para_mmse["ksi_min"] = 10 ** (-25 / 10)   # ksi最小值 -25dB
    
    para_mmse["mu_VAD"] = 0.98 # VAD噪声跟踪的参数
    para_mmse["th_VAD"] = 3  # VAD 判定阈值 3db
    
    para_mmse["fun_GAN"] = Gan_sqr_mmse
    # mmse 增强
    enhance = enh_mmse(noisy,noisy[:1000],para_mmse)
    
    sf.write("enhce_sqr_mmse.wav",enhance,fs)

    
    plt.subplot(3,1,1)
    plt.specgram(clean,NFFT=256,Fs=fs)
    plt.xlabel("clean specgram")
    plt.subplot(3,1,2)
    plt.specgram(noisy,NFFT=256,Fs=fs)
    plt.xlabel("noisy specgram")   
    plt.subplot(3,1,3)
    plt.specgram(enhance,NFFT=256,Fs=fs)
    plt.xlabel("enhece specgram")  
    plt.show()
    
    
    