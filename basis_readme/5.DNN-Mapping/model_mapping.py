import torch
import torch.nn as nn
from hparams import hparams

class DNN_Mapping(nn.Module):
    def __init__(self,para):
        super(DNN_Mapping,self).__init__()
        self.dim_in = para.dim_in
        self.dim_out = para.dim_out
        self.dim_embeding = para.dim_embeding
        self.dropout = para.dropout
        self.negative_slope = para.negative_slope
        
        self.BNlayer = nn.BatchNorm1d(self.dim_out)
        
        self.model = nn.Sequential(
                        # 先行正则化
                        nn.BatchNorm1d(self.dim_in),

                        # 第一层
                        nn.Linear(self.dim_in, self.dim_embeding),
                        nn.BatchNorm1d(self.dim_embeding),
                        # nn.ReLU(),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Dropout(self.dropout),
                        
                        # 第二层
                        nn.Linear(self.dim_embeding, self.dim_embeding),
                        nn.BatchNorm1d(self.dim_embeding),
                        # nn.ReLU(),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Dropout(self.dropout),
                        
                        # 第三层
                        nn.Linear(self.dim_embeding, self.dim_embeding),
                        nn.BatchNorm1d(self.dim_embeding),
                        # nn.ReLU(),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Dropout(self.dropout),
                        
                        # 第四层
                        nn.Linear(self.dim_embeding, self.dim_out),
                        nn.BatchNorm1d(self.dim_out),
                        
                        )
                        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
               
            
    def forward(self,x,y=None, istraining = True):
        out_enh = self.model(x)
        if istraining:
            out_targrt = self.BNlayer(y)
            return out_enh,out_targrt
        else:
            return out_enh
        
if __name__ == "__main__":
    para = hparams()
    m_model = DNN_Mapping(para)
    print(m_model)
    x = torch.randn(3,para.dim_in)
    y = m_model(x)
    print(y.shape)

        
        