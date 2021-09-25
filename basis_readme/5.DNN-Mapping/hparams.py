import torch
class hparams():
    def __init__(self):
        self.file_scp = "scp/train_DNN_enh.scp"
        
        self.para_stft = {}
        self.para_stft["N_fft"] = 512
        self.para_stft["win_length"] = 512
        self.para_stft["hop_length"] = 128
        self.para_stft["window"] = 'hamming'
       
        self.n_expand = 3
        self.dim_in = int((self.para_stft["N_fft"]/2 +1)*(2*self.n_expand+1))
        self.dim_out = int((self.para_stft["N_fft"]/2 +1))
        self.dim_embeding = 2048
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.negative_slope = 1e-4
        self.dropout = 0.1
        
        