import torch


class hparams():
    def __init__(self, B=6, blocknums=10, modeltype="TGSA"):
        self.sr = 16000
        self.B = B
        #self.gpunum = 2
        # self.batch_size = self.gpunum*self.B  # 用DP来做多卡训练
        self.batch_size = self.B  # ddp来做多卡训练 或者单卡训练
        self.n_fft = 256
        self.D = 129  # ( 要由 n_fft 算)

        '''

        '''

        # train
        self.initlr = 1e-4
        self.total_epoch = 120
        self.save_inter = 10
        self.logfilename = "./" + modeltype + "_train_v10.txt"
        self.testlogfilename = "./" + modeltype + "_test_v10.txt"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.divide_radius = 1.5 # 取语音半径
        self.max_time_len = 5.0

        # models structure
        self.init_gaussion_theta: float = 1.5
        self.head_nums = 3
        self.D_E = self.D // self.head_nums  # head size
        self.E = self.head_nums
        self.is_punish = True

        self.block_nums = blocknums  # 6或者10
        self.modeltype: str = modeltype

        # loss
        self.aa = torch.tensor(2.0)  # SDR loss

        # dataset & eval  & test
        self.iseval = False
        self.eval_interval = 100# 每多少个step eval一次数据集sdr。

        self.trainscpdir = '/data/private/user17/workspace/ywh/datasets/voicebank/se-train-16k.txt'
        self.testscpdir = '/data/private/user17/workspace/ywh/datasets/voicebank/se-test-16k.txt'
        self.rootdir = '/data/private/user17/workspace/ywh/datasets/voicebank'


        self.eval_wavs_dir = '/data/private/user17/workspace/ywh/datasets/voicebank/eval_wavs'
if __name__=="__main__":
    import os
    epoch =1
    para = hparams()
    print("./checkpoints/"+"param_{}_".format((epoch + 1) // para.save_inter) +os.path.basename(para.logfilename).split('.')[0]+'.pt')
