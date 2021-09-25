import numpy as np
import librosa
import os

def wav_split(wav,win_length,strid):
    slices = []
    if len(wav)> win_length:
    
        for idx_end in range(win_length, len(wav), strid):
            idx_start = idx_end - win_length
            slice_wav = wav[idx_start:idx_end]
            slices.append(slice_wav)

        # 拼接最后一帧
        slices.append(wav[-win_length:])
    return slices
    
    
def save_slices(slices,name):
    
    name_list = []
    if len(slices) >0:
        for i , slice_wav in enumerate(slices):
            name_slice = name+"_"+str(i)+'.npy'
            np.save(name_slice,slice_wav)
            name_list.append(name_slice)
    return name_list
    
    
    

if __name__ == "__main__":
    clean_wav_path = "/home/sdy/dataset/enh_voicebank/clean_trainset_wav"
    noisy_wav_path = "/home/sdy/dataset/enh_voicebank/noisy_trainset_wav/"
    
    catch_train_clean = '/home/sdy/ctach_segan/clean'
    catch_train_noisy = '/home/sdy/ctach_segan/noisy'
    
    os.makedirs(catch_train_clean,exist_ok=True)
    os.makedirs(catch_train_noisy,exist_ok=True)
    
    win_length = 16384
    strid = int(win_length/2)
    # 遍历所有wav文件
    with open("scp/train_segan.scp",'wt') as f:
        for root, dirs, files in os.walk(clean_wav_path):
            for file in files:
                file_clean_name = os.path.join(root,file)
                name = os.path.split(file_clean_name)[-1]
                if name.endswith("wav"):
                    
                    file_noisy_name = os.path.join(noisy_wav_path,name)
                    print("processing file %s"%(file_clean_name))
                    
                    if not os.path.exists(file_noisy_name):
                        print("can not find file %s"%(file_noisy_name))
                        continue
                    
                    clean_data,sr = librosa.load(file_clean_name,sr=16000,mono=True)
                    noisy_data,sr = librosa.load(file_noisy_name,sr=16000,mono=True)
                    
                    if not len(clean_data) == len(noisy_data):
                        print("file length are not equal")
                        continue
                    # 干净语音分段+保存
                    clean_slices = wav_split(clean_data,win_length,strid)
                    clean_namelist = save_slices(clean_slices,os.path.join(catch_train_clean,name))
                    
                    # 噪声语音分段+保存
                    noisy_slices = wav_split(noisy_data,win_length,strid)
                    noisy_namelist = save_slices(noisy_slices,os.path.join(catch_train_noisy,name))
                    
                    for clean_catch_name,noisy_catch_name in zip(clean_namelist,noisy_namelist):
                        f.write("%s %s\n"%(clean_catch_name,noisy_catch_name))
                    
                        
                    
                    
                    
                    
                    