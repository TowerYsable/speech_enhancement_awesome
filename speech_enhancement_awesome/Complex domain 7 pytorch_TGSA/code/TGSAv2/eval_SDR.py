
# encoding: utf-8
import torch
import os
import torchvision


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(10)



from models import TGSAmodel
from data_prepare import get_Loader
from utils import wavdir_2_padded_features
from hparams import hparams


# 单机单卡
para = hparams()
para.iseval = True
device = para.device

totalloss = 0
step = 0
logfile =  para.testlogfilename  # 注意用 test

if os.path.exists(logfile):
    os.remove(logfile)

with open(logfile, 'a', encoding='utf-8') as f:
    f.write(logfile.split('.')[0])
    f.write(str(para.__dict__))
    f.write("\n")
    f.close()


root = para.rootdir
scpdir = para.testscpdir   # 注意用 test

# 定义 dataset loader
train_loader = get_Loader(scpdir,root,para.batch_size)

GSAmodel = TGSAmodel(para)
model_dict_dir = './checkpoints/param_2_TGSA_train_v10.pt'
GSAmodel.load_state_dict(torch.load(model_dict_dir))

GSAmodel = GSAmodel.to(device)
iternum=0
for batch in train_loader:
    iternum += 1
    clean_wavdirs,noisy_wavdirs = batch

    pd_X,pd_Y ,seq_len,cleanwavforms= wavdir_2_padded_features(noisy_wavdirs,clean_wavdirs,para)

    pd_X = pd_X.to(device)
    pd_Y = pd_Y.to(device)

    pred_mask = GSAmodel(pd_X, seq_len)
    pred_mask = pred_mask.to(device)
    #print(pred_mask.shape) # [frames,fft dim]


    for i in range(pred_mask.shape[0]):
        torchvision.utils.save_image(pred_mask[i,:seq_len[i],:].unsqueeze(0).unsqueeze(0).data,'./test_pred_wavs/{}.bmp'.format(i + 1), padding=0)
    print("写入一个batch 的mask图片完毕")
    exit()

    batchloss = GSAmodel.get_loss(pred_mask, pd_X, pd_Y, seq_len,cleanwavforms,para)

    sdrloss = -batchloss
    print(sdrloss)
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write("step:{},sdrvalue:{} ".format(iternum,sdrloss))
        f.write("\n")
        f.close()
