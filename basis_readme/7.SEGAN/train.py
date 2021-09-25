import torch
from dataset import SEGAN_Dataset,emphasis
from hparams import hparams
from model import Generator, Discriminator
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.autograd import Variable
if __name__ == "__main__":
    
   
    
    # 定义device
    device = torch.device("cuda:1")
    
    # 导入参数
    para = hparams()
    
    # 创建数据保存文件夹
    os.makedirs(para.save_path,exist_ok=True)
    
    # 创建生成器
    generator = Generator()
    generator = generator.to(device)
    
    # 创建鉴别器
    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    
    # # 创建G 的优化器
    # m_G_optimizer = torch.optim.RMSprop(m_G.parameters(), lr=para.lr_G)
    
    # # 创建D 的优化器
    # d_optimizer = torch.optim.RMSprop(m_D.parameters(), lr=para.lr_D)
    # optimizers
    
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)
    
    # 定义数据集
    m_dataset = SEGAN_Dataset(para)
    
    # 获取ref-batch
    ref_batch = m_dataset.ref_batch(para.ref_batch_size)
    ref_batch = Variable(ref_batch)
    ref_batch = ref_batch.to(device)
    
    # 定义dataloader
    m_dataloader = DataLoader(m_dataset,batch_size = para.batch_size,shuffle = True, num_workers = 8)
    loss_d_all =0
    loss_g_all =0
    n_step =0
    for epoch in range(para.n_epoch):
        
        for i_batch, sample_batch in enumerate(m_dataloader):
            batch_clean = sample_batch[0]
            batch_noisy = sample_batch[1]
            batch_clean = Variable(batch_clean)
            batch_noisy = Variable(batch_noisy)
            
            batch_clean = batch_clean.to(device)
            batch_noisy = batch_noisy.to(device)
            
            # print(batch_clean.size())
            # print(batch_noisy.size())
            
            batch_z = nn.init.normal(torch.Tensor(batch_clean.size(0), para.size_z[0], para.size_z[1]))
            batch_z = Variable(batch_z)
            batch_z = batch_z.to(device)
            
            
            discriminator.zero_grad()
            train_batch = Variable(torch.cat([batch_clean,batch_noisy],axis=1))
            # train_batch = torch.cat([batch_clean,batch_noisy],axis=1)
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            # clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(batch_noisy, batch_z)
            outputs = discriminator(torch.cat((generated_outputs, batch_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            # noisy_loss.backward()

            d_loss = clean_loss + noisy_loss
            d_loss.backward()
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(batch_noisy, batch_z)
            gen_noise_pair = torch.cat((generated_outputs, batch_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(batch_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()
            
            
            
            # # 更新参数 D
            # m_D.zero_grad()
            
            # # 计算真实数据的 d_loss
            # batch_train_real = torch.cat((batch_clean,batch_noisy),axis=1)
  
            
            # real_d_out = m_D(batch_train_real,batch_ref)
            # d_real_loss = torch.mean((real_d_out-1.0)**2)
            # d_real_loss.backward()
            
            # # 计算虚假数据的 d_loss
            # batch_g = m_G(batch_noisy,batch_z)
            # batch_train_fake = torch.cat((batch_g,batch_noisy),axis=1)
            # fake_d_out = m_D(batch_train_fake,batch_ref)
            # d_fake_loss = torch.mean(fake_d_out**2)
            # d_fake_loss.backward()
            
            # d_loss =  d_real_loss + d_fake_loss
            
            # # d_loss.backward()
            # m_D_optimizer.step()
            
            # # 更新参数G
            # m_G.zero_grad()
            
            # batch_g = m_G(batch_noisy,batch_z)
            
            # batch_train_fake = torch.cat((batch_g,batch_noisy),axis=1)
            # fake_d_out = m_D(batch_train_fake,batch_ref)
            # # D 将生成语音认作真实语音
            # g_loss_1 = 0.5*torch.mean((fake_d_out-1.0)**2)
            # # 生成语音和真实语音的L1距离最小
           
            # L1_dis = torch.abs(torch.add(batch_g, torch.neg(batch_clean)))
            # g_loss_2 = 100*torch.mean(L1_dis)
            
            # g_loss = g_loss_1 + g_loss_2
            # g_loss.backward()
            # m_G_optimizer.step()
            
            # loss_d_all = loss_d_all+d_loss
            # loss_g_all = loss_g_all+g_loss
            # n_step = n_step+1
            # print("epoch = %d step=%d loss_d = %.5f loss_g = %.5f  loss_g1= %.5f  loss_g2=%.5f"%(epoch,i_batch,d_loss,g_loss,g_loss_1,g_loss_2))
            print("Epoch %d:%d d_clean_loss %.4f, d_noisy_loss %.4f, g_loss %.4f, g_conditional_loss %.4f"%(epoch + 1,i_batch,clean_loss,noisy_loss,g_loss,g_cond_loss))
        
        
        # mean_d = loss_d_all/n_step
        # mean_g = loss_g_all/n_step
        g_model_name = os.path.join(para.path_save,"G_"+str(epoch)+"_%.4f"%(g_cond_loss)+".pkl")
        d_model_name = os.path.join(para.path_save,"D_"+str(epoch)+"_%.4f"%(noisy_loss)+".pkl")
        torch.save(generator.state_dict(), g_model_name)
        torch.save(discriminator.state_dict(), d_model_name)

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
    
    
    
    
    
    
    
    