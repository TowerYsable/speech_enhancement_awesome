import torch
import torch.nn as nn
from hparams import hparams
from torch.utils.data import Dataset,DataLoader
from dataset import TIMIT_Dataset,my_collect
from model_mapping import DNN_Mapping
import os

if __name__ == "__main__":
    
    # 定义device
    device = torch.device("cuda:0")
    
    # 获取模型参数
    para = hparams()
    
    # 定义模型
    m_model = DNN_Mapping(para)
    m_model = m_model.to(device)
    m_model.train()
    
    # 定义损失函数
    loss_fun = nn.MSELoss()
    # loss_fun = nn.L1Loss()
    loss_fun = loss_fun.to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(
        params=m_model.parameters(),
        lr=para.learning_rate)
    
    # 定义数据集
    m_Dataset= TIMIT_Dataset(para)
    m_DataLoader = DataLoader(m_Dataset,batch_size = para.batch_size,shuffle = True, num_workers = 4, collate_fn = my_collect)
    
    # 定义训练的轮次 
    n_epoch = 100
    n_step = 0
    loss_total = 0
    for epoch in range(n_epoch):
        # 遍历dataset中的数据
        for i_batch, sample_batch in enumerate(m_DataLoader):
            train_X = sample_batch[0]
            train_Y = sample_batch[1]
            
            train_X = train_X.to(device)
            train_Y = train_Y.to(device)
            
            m_model.zero_grad()
            # 得到网络输出
            output_enh,out_target = m_model(x=train_X,y=train_Y)
            
            # 计算损失函数
            loss = loss_fun(output_enh,out_target)
            
            # 误差反向传播
            # optimizer.zero_grad()
            loss.backward()
            
            # 进行参数更新
            # optimizer.zero_grad()
            optimizer.step()
            
            n_step = n_step+1
            loss_total = loss_total+loss
            
            # 每100 step 输出一次中间结果
            if n_step %100 == 0:
                print("epoch = %02d  step = %04d  loss = %.4f"%(epoch,n_step,loss))
        
        # 训练结束一个epoch 计算一次平均结果
        loss_mean = loss_total/n_step
        print("epoch = %02d mean_loss = %f"%(epoch,loss_mean))
        loss_total = 0
        n_step =0
        
        # 进行模型保存
        save_name = os.path.join('save','model_%d_%.4f.pth'%(epoch,loss_mean))
        torch.save(m_model,save_name)
        
                
            
            
            
            
            
        
        
        
        
        
        