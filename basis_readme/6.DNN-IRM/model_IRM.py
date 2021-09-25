import torch
import torch.nn as nn
from hparams import hparams

class DNN_IRM(nn.Module):
    def __init__(self,para):
        super(DNN_IRM,self).__init__()
        self.dim_in = para.dim_in
        self.dim_out = para.dim_out
        self.dim_embeding = para.dim_embeding
        self.dropout = para.dropout
        self.negative_slope = para.negative_slope
        
        self.model = nn.Sequential(
                        nn.BatchNorm1d(self.dim_in),                        
                        # 第一层
                        nn.Linear(self.dim_in, self.dim_embeding),
                        nn.BatchNorm1d(self.dim_embeding),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Dropout(self.dropout),
                        
                        # 第二层
                        nn.Linear(self.dim_embeding, self.dim_embeding),
                        nn.BatchNorm1d(self.dim_embeding),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Dropout(self.dropout),
                        
                        # 第三层
                        nn.Linear(self.dim_embeding, self.dim_embeding),
                        nn.BatchNorm1d(self.dim_embeding),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Dropout(self.dropout),
                        
                        # 第四层
                        nn.Linear(self.dim_embeding, self.dim_out),
                        nn.BatchNorm1d(self.dim_out),
                        nn.LeakyReLU(self.negative_slope),
                        nn.Sigmoid()
                        )
                        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
               
            
    def forward(self,x):
        out_mask = self.model(x)
        return out_mask
        
if __name__ == "__main__":
    para = hparams()
    m_model = DNN_IRM(para)
    print(m_model)
    x = torch.randn(3,para.dim_in)
    y = m_model(x)
    print(y.shape)

        
        
