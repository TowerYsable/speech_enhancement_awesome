from torch.utils.data import Dataset, DataLoader
'''
'/home/ywh/voicebank/se-train-16k.txt'
'/home/ywh/voicebank/se-test-16k.txt'
'''
class data_scp(Dataset):

    def __init__(self,filedir,root):
        # 使用2个列表分别保存 noisy语音和 clean语音的路径。
        self.clean_scplist = []
        self.noisy_scplist = []
        with open(filedir,encoding='utf-8') as f:
            for line in f.readlines():
                self.clean_scplist.append(root+'/'+line.split()[0])
                self.noisy_scplist.append(root+'/'+line.split()[1])
        f.close()
        self.L = len(self.clean_scplist)

    def __getitem__(self, index):
        return self.clean_scplist[index],self.noisy_scplist[index]

    def __len__(self):
        return self.L

def get_Loader(filedir,root,batchsize):
    return DataLoader(data_scp(filedir,root),batch_size=batchsize,shuffle=True,drop_last=True)




if __name__=="__main__":
    loader = get_Loader('/home/ywh/voicebank/se-train-16k.txt','/home/ywh/voicebank',2)
    print(len(loader.dataset))
    batch = next(iter(loader))
    cd,nd = batch
    print(cd)
    print(nd)




