from matplotlib import pyplot as plt
import torch
import re
import numpy as np
import torchaudio
def plotit(waveform):
    waveform = waveform.cpu()
    L = waveform.shape[-1]
    w = waveform.detach().numpy()
    L = np.array([range(L)])
    plt.plot(L,w)
    plt.show()

def plot_trainlog(trainlogfiledir:str,subject:str): #.选择 log文件和 要画的 属性值。
    #   第0 - 6个数字为 epoch 、step、batchloss、avgloss、loss1、loss2、avgacc。
    valuedic = {'epoch':0,'step':1,'batchloss':2,'avgloss':3,'SDRloss':4}
    values = []


    flag = 0
    for k,v in valuedic.items():
        if k == subject:
            flag = v

    if subject != "eval_avgSDR":
        with open(trainlogfiledir, encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                line = line.strip('\n').strip(' ')
                if line[:6] == "train,":
                    numslist = line.split(',')[1:-1]
                    nums = [float(k.split(':')[1]) for k in numslist]
                    values.append(nums[flag])
            f.close
    else:
        with open(trainlogfiledir, encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                line = line.strip('\n').strip(' ')
                if line[:5] == "eval,":
                    values.append(float(str(line.split(',')[-1].split(':')[-1])))
            f.close


    tail_ratio = 1
    stat_ratio = 1-tail_ratio
    L = len(values)
    stat_L =int(L*stat_ratio)
    plt.plot(range(len(values))[stat_L:L] ,values[stat_L:L] )
    plt.ylabel(subject)
    plt.show()

    print("avg is ",sum(values)/len(values))






if __name__=="__main__":
    f1 = r'D:\mything\myGSAcheck\user17\v10_Adam\TGSA_train_v10.txt'
    plot_trainlog(f1, 'batchloss')
    plot_trainlog(f1, 'avgloss')
    plot_trainlog(f1, 'SDRloss')
    plot_trainlog(f1, 'eval_avgSDR')

