import torch

if __name__ == "__main__":
    model_name = "save/model_2_0.0018.pth"
    m_model = torch.load(model_name,map_location = torch.device('cpu'))
    m_model.eval()
    
    model_dic = m_model.state_dict()
    
    for k,v in model_dic.items():
        print('k:'+k)
        print(v.size())
        
    print(model_dic['BNlayer.weight'].data)