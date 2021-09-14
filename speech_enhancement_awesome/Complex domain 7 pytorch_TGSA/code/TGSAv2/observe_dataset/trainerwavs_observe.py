import torch
import torchaudio
import numpy
import os
from matplotlib import pyplot as plt





def obser_time():
    traintxt = r'D:\mything\myGSA\训练信息\voicebank\se-train-16k.txt'
    testtxt = r'D:\mything\myGSA\训练信息\voicebank\se-test-16k.txt'

    root = r'D:\mything\myGSA\训练信息\voicebank'
    # 观测数据集语音长度。
    train_timelen = []
    test_timelen = []

    train_signallen = []
    test_signallen = []
    with open(traintxt,encoding='utf-8')as f1 :
        for line in f1.readlines():
            cleanwi,sr = torchaudio.load(os.path.join(root,line.split()[0]))
            # nosiywi, sr = torchaudio.load(os.path.join(root, line.split()[1]))
            # print(cleanwi.shape)
            # print(nosiywi.shape)
            # exit()
            train_timelen.append(cleanwi.shape[-1]/sr) # 时间
            train_signallen.append(cleanwi.shape[-1] )  # 采样点数

        f1.close()
    with open(testtxt,encoding='utf-8')as f2 :
        for line in f2.readlines():
            cleanwi,sr = torchaudio.load(os.path.join(root,line.split()[0]))
            # nosiywi, sr = torchaudio.load(os.path.join(root, line.split()[1]))
            # print(cleanwi.shape)
            # print(nosiywi.shape)
            # exit()
            test_timelen.append(cleanwi.shape[-1]/sr)
            test_signallen.append(cleanwi.shape[-1])
        f2.close()
    numpy.save('./trainwavs_timelen',train_timelen)
    numpy.save('./trainwavs_signallen',train_signallen)
    numpy.save('./testwavs_timelen',test_timelen)
    numpy.save('./testwavs_signallen',test_signallen)


def obser_time_distribution():
    train_timelen = numpy.load('trainwavs_timelen.npy',)
    test_timelen = numpy.load('testwavs_timelen.npy')

    # 观察平均、最大、最小、分布
    avg_tr = sum(train_timelen)/len(train_timelen)
    max_tr = max(train_timelen)
    min_tr = min(train_timelen)
    print("train: avg :{},  max:{}  , min :{} ，totalnumber: {}".format(avg_tr,max_tr,min_tr,len(train_timelen)))
    plt.hist(train_timelen,bins=10)
    plt.show()

    # 观察平均、最大、最小、分布
    avg_te = sum(test_timelen)/len(test_timelen)
    max_te = max(test_timelen)
    min_te = min(test_timelen)
    print("test: avg :{},  max:{}  , min :{} ，totalnumber: {}".format(avg_te,max_te,min_te,len(test_timelen)))
    plt.hist(test_timelen,bins=10)
    plt.show()






if __name__=="__main__":
    # obser_time()
    #obser_time_distribution()
    pass






