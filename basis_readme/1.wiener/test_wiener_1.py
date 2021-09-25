import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


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
    clean_wav_file = "sf1_cln.wav"
    clean,fs = librosa.load(clean_wav_file,sr=None) 
    
    # 读取读取噪声语音
    noisy_wav_file = "sf1_n0L.wav"
    noisy,fs = librosa.load(noisy_wav_file,sr=None)
    
    # 获取噪声
    noise = noisy-clean
    
    # 设置模型参数
    para_wiener = {}
    para_wiener["n_fft"] = 256
    para_wiener["hop_length"] = 128
    para_wiener["win_length"] = 256
    para_wiener["alpha"] = 1
    para_wiener["beta"] = 8
    
    # 维纳滤波
    H,enhenc = wiener_filter(noisy,clean,noise,para_wiener)
    
    sf.write("enhce.wav",enhenc,fs)

    
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
    
    
    