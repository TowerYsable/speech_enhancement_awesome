# encoding: utf-8
import torch
import os
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import TGSAmodel
from data_prepare import get_Loader
from utils import wavdir_2_padded_features,ob_mask_distribution
from hparams import hparams

import random
import numpy as np


# 定义网络
class trainer(nn.Module):
    def __init__(self, para:hparams):
        super(trainer, self).__init__()
        self.para = para
        self.device = para.device
        # 定义 dataset loader
        self.train_loader = get_Loader(para.trainscpdir, self.para.rootdir, para.batch_size)
        self.valid_loader = get_Loader(para.testscpdir, self.para.rootdir, para.batch_size)
        self.GSAmodel = TGSAmodel(para)
        #self.optimizer_model = torch.optim.SGD(GSAmodel.parameters(), lr=para.initlr, momentum=0.9, dampening=0, weight_decay=0.0005, nesterov=False)

        self.optimizer_model = torch.optim.Adam(self.GSAmodel.parameters(), lr=para.initlr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00005)
        self.sheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='max', factor=0.9, patience=10, verbose=False,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7,
                                                   eps=1e-08)

        #self.sheduler_model = torch.optim.lr_scheduler.StepLR(self.optimizer_model, 10, gamma=0.9, last_epoch=-1)

        ## logger
        self.logfile = para.logfilename # log文件路径
        self.batch_nums = int(len(self.train_loader.dataset) / para.batch_size) #每个epoch 的 batch数量。
        self.present_avg_eval_sdr = 0 # 当前step检测的测试集sdr
        self.step_num = 0  # 第i个epcoh的时候，模型 已经走过的 batch step的数量。
        self.totalloss = 0 # 总loss
        self.batch_loss = 0  # 当前steploss
        self.steploss = 0  # 总loss/ stepnum


        pass

    def init_logger(self):
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        with open(self.logfile, 'a', encoding='utf-8') as f:
            f.write(self.logfile.split('.')[0])
            # f.write(str(self.para.__dict__) + "\n")
            for item in para.__dict__.items():
                f.write("{} : {}\n".format(str(item[0]),str(item[1])))
            f.write(str(type(self.optimizer_model)) + "\n")
            f.write(str(type(self.sheduler_model)) + "\n")
            f.write("------all hparams writed------\n")
            f.write("------------------------------\n")
            f.close()


    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def eval_SDR(self):

        # 每隔parade.eval_interval 就测试一下 测试集的平均SDR
        self.GSAmodel.eval()
        with torch.no_grad():
            eval_sdrs = []
            for val_batch in self.valid_loader:
                val_clean_dir, val_noisy_dir = val_batch
                pd_X, pd_Y, seq_len, cleanwavforms = wavdir_2_padded_features(val_noisy_dir, val_clean_dir, self.para)
                pd_X = pd_X.to(self.device)
                pd_Y = pd_Y.to(self.device)
                pred_mask = self.GSAmodel(pd_X, seq_len)
                pred_mask = pred_mask.to(self.device)
                batchloss = self.GSAmodel.get_loss(pred_mask, pd_X, pd_Y, seq_len, cleanwavforms, self.para, val_clean_dir)
                eval_sdrs.append(-batchloss)
        return sum(eval_sdrs)/len(eval_sdrs)

    def train_a_epoch(self,epoch):
        for batch in self.train_loader:
            self.GSAmodel.train()
            self.step_num += 1
            is_param_nan = False # 标识当前batch是 否出现了非数的参数更新
            clean_wavdirs, noisy_wavdirs = batch
            pd_X, pd_Y, seq_len, cleanwavforms = wavdir_2_padded_features(noisy_wavdirs, clean_wavdirs, self.para)
            pd_X = pd_X.to(self.device)
            pd_Y = pd_Y.to(self.device)
            pred_mask = self.GSAmodel(pd_X, seq_len)
            pred_mask = pred_mask.to(self.device)
            batchloss = self.GSAmodel.get_loss(pred_mask, pd_X, pd_Y, seq_len, cleanwavforms, self.para, clean_wavdirs)
            sdrloss = -batchloss
            self.optimizer_model.zero_grad()
            batchloss.backward()
            d, a = ob_mask_distribution(torch.abs(pred_mask))
            # 万一梯度爆炸为NAN，则直接跳过该batch。
            for p in self.GSAmodel.parameters():
                if torch.any(torch.isnan(p.grad)):
                    is_param_nan = True
                    break
            if is_param_nan:
                continue
            #-------------------------------------------------------
            clip_grad_norm_(self.GSAmodel.parameters(), max_norm=20, norm_type=2)
            self.optimizer_model.step()
            self.step_num += 1
            self.totalloss += batchloss
            steploss = self.totalloss / self.step_num

            print("MASK:distribution: {} , avg : {} , paranan: {}".format(d, a, is_param_nan))
            print("train,epoch:{} ,batch_step:{} ,batchloss:{} ,avg_loss:{} ,sdrloss:{}".format \
                      (epoch, self.step_num % (self.batch_nums), batchloss.item(), steploss.item(), sdrloss.item()))
            print("---------------------------------------------------------------------------------")
            with open(self.logfile, 'a', encoding='utf-8') as f:
                f.write("train,epoch:{} ,batch_step:{} ,batchloss:{} ,avg_loss:{} ,sdrloss:{} ,pnan:{}\n ".format \
                            (epoch,self.step_num % (self.batch_nums), batchloss.item(), steploss.item(), sdrloss.item(),
                             is_param_nan))
                f.close()

            ####  eval
            if self.step_num % self.para.eval_interval == 0:
                avg_sdr = self.eval_SDR()
                self.present_avg_eval_sdr = avg_sdr
                with open(self.logfile, 'a', encoding='utf-8') as f:
                    f.write("eval,epoch:{} ,batch_step:{} ,eval_avgSDR:{}\n ".format \
                                (epoch, self.step_num % (self.batch_nums),avg_sdr.item()))
                    f.close()


            # if self.step_num % 5 == 0:
            #     exit()


        self.sheduler_model.step(self.present_avg_eval_sdr)


    def forward(self):
        self.init_logger()
        for epoch in range(self.para.total_epoch):
            self.train_a_epoch(epoch)
            # 每10个epoch存储模型
            if (epoch + 1) % self.para.save_inter == 0:
                torch.save(self.GSAmodel.state_dict(),
                           "./checkpoints/" + "param_{}_".format((epoch + 1) // self.para.save_inter) + \
                           os.path.basename(self.para.logfilename).split('.')[0] + '.pt')
        pass

if __name__=="__main__":
    para = hparams()
    trainer_v09 = trainer(para)
    trainer_v09 = trainer_v09.to(para.device)
    trainer_v09.setup_seed(10)
    trainer_v09.forward()
