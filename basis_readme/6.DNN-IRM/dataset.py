import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from hparams import hparams
import librosa
import random
import soundfile as sf



def feature_stft(wav,para):
    spec = librosa.stft(wav,
                      n_fft=para["N_fft"],
                      win_length = para["win_length"],
                      hop_length = para["hop_length"],
                      window =para["window"])
                      
    mag =   np.abs(spec)
    phase = np.angle(spec)    
        
    return mag.T, phase.T    #  T x D

# feature T x D
# out   T x D*(2*expand+1)
def feature_contex(feature,expend):
    feature = feature.unfold(0,2*expend+1,1)  # T x D x  2*expand+1
    feature = feature.transpose(1,2)           # T x  2*n_expand+1  x D 
    feature = feature.view([-1,(2*expend+1)*feature.shape[-1]]) # T x  D * 2*n_expand+1
    return feature
    
    
def get_mask(clean,noisy,para):
    noise = noisy-clean
    
    clean_mag,_ =  feature_stft(clean,para)
    noisy_mag,_ =  feature_stft(noisy,para)
    noise_mag,_ =  feature_stft(noise,para)
    
    mask =   (clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2))**(0.5)
    return clean_mag,noisy_mag,mask
    
    
    
    

class TIMIT_Dataset(Dataset):
    
    def __init__(self,para):
        
        self.file_scp = para.file_scp
        self.para_stft = para.para_stft
        self.n_expand = para.n_expand
        
        files = np.loadtxt(self.file_scp,dtype = 'str')
        self.clean_files = files[:,1].tolist()
        self.noisy_files = files[:,0].tolist()
         
        print(len(self.clean_files))
    
    def __len__(self):
        return len(self.clean_files)
        
    
    def __getitem__(self,idx):
        
        # 读取干净语音
        clean_wav,fs = sf.read(self.clean_files[idx],dtype = 'float32')
        clean_wav = clean_wav.astype('float32')
        
        #  读取含噪语音
        noisy_wav,fs = sf.read(self.noisy_files[idx],dtype = 'float32')
        noisy_wav = noisy_wav.astype('float32')
        
        # 进行 特征提取
        clean_mag,noisy_mag,mask = get_mask(clean_wav,noisy_wav,self.para_stft)
        
        # 转为torch格式
        X_train = torch.from_numpy(np.log(noisy_mag**2))
        Y_train = torch.from_numpy(mask)
        
        # 拼帧
        X_train = feature_contex(X_train,self.n_expand)
        Y_train = Y_train[self.n_expand:-self.n_expand,:]
        return X_train, Y_train

def my_collect(batch):
    batch_X = [item[0] for item in batch]
    batch_Y = [item[1] for item in batch]
    batch_X = torch.cat(batch_X,0)
    batch_Y = torch.cat(batch_Y,0)
    return[batch_X.float(),batch_Y.float()]
    
    
if __name__ == '__main__':
    
    # 数据加载测试
    para = hparams()
    
    m_Dataset= TIMIT_Dataset(para)
    
    m_DataLoader = DataLoader(m_Dataset,batch_size = 2,shuffle = True, num_workers = 4, collate_fn = my_collect)
    
    for i_batch, sample_batch in enumerate(m_DataLoader):
        train_X = sample_batch[0]
        train_Y = sample_batch[1]
        print(train_X.shape)
        print(train_Y.shape)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
