import torch
class hparams():
    def __init__(self):
        self.train_scp = "scp/train_segan.scp"
        
        self.fs = 16000
        self.win_len = 16384
        
        self.ref_batch_size = 400
        
        self.lr_G = 2e-4
        self.lr_D = 2e-4
        
        self.batch_size = 128
        self.n_epoch =100
        
        self.size_z = (1024,8)
        
        self.w_g_loss1 = 0.5
        self.w_g_loss2 = 2
        
        self.save_path = "save"
        self.n_epoch = 90
     
        self.ref_batch_size=128
        self.path_save ='save'
