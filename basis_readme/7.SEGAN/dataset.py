import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from hparams import hparams
import librosa
import random
import soundfile as sf

# 预加重与反预加重
def emphasis(signal, emph_coeff=0.95, pre=True):
    
    if pre:
        result = np.append(signal[0], signal[1:] - emph_coeff * signal[:-1])
    else:
        result = np.append(signal[0], signal[1:] + emph_coeff * signal[:-1])
    
    return result


class SEGAN_Dataset(Dataset):
    
    def __init__(self,para):
        
        self.file_scp = para.train_scp
        
        files = np.loadtxt(self.file_scp,dtype = 'str')
        self.clean_files = files[:,0].tolist()
        self.noisy_files = files[:,1].tolist()
    
    def __len__(self):
        return len(self.clean_files)
        
    def __getitem__(self,idx):
        
        # 读取干净语音并预加重
        clean_wav = np.load(self.clean_files[idx])
        clean_wav = emphasis(clean_wav)
        # 读取含噪语音
        noisy_wav = np.load(self.noisy_files[idx])
        noisy_wav = emphasis(noisy_wav)
        
        # 读取干净语音并预加重
        clean_wav = torch.from_numpy(clean_wav)
        noisy_wav = torch.from_numpy(noisy_wav)
        
        # 增加一个维度
        clean_wav = clean_wav.reshape(1,-1)
        noisy_wav = noisy_wav.reshape(1,-1)
        
        return clean_wav, noisy_wav
    
    def ref_batch(self,batch_size):
        
        index = np.random.choice(len(self.clean_files),batch_size).tolist()
        
        catch_clean = [emphasis(np.load(self.clean_files[i])) for i in index]
        catch_noisy = [emphasis(np.load(self.noisy_files[i])) for i in index]
        catch_clean = np.expand_dims(np.array(catch_clean),axis=1)
        catch_noisy = np.expand_dims(np.array(catch_noisy),axis=1)
           
        batch_wav = np.concatenate((catch_clean,catch_noisy),axis=1)
        return torch.from_numpy(batch_wav)
       
        
        
        
    
    

if __name__ == "__main__":
    para = hparams()
    
    m_Dataset= SEGAN_Dataset(para)
    
    m_DataLoader = DataLoader(m_Dataset,batch_size = 3,shuffle = True, num_workers = 4)
    
    # m_bath_ref = m_Dataset.ref_batch(32)
    # print(m_bath_ref.size())
    
    
    
    for i_batch, sample_batch in enumerate(m_DataLoader):
        batch_clean = sample_batch[0]
        batch_noisy = sample_batch[1]
        print(batch_clean[1])
        print(batch_noisy[1])
    
    
    
    