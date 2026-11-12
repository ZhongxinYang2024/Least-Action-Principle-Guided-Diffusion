import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
import os
import re

def infinite_iterator(data_source):
    while True:  # 外层无限循环
        for item in data_source:  # 内层遍历数据集
            yield item
    
# 自定义数据集类
class CustomDataset_SIN(Dataset):
    def __init__(self, resolution=128):
        super().__init__()
        # ######################
        num_samples = 10000;
        num_T = 2; 
        # ######################
        data = torch.zeros(num_samples, resolution)
        labels = torch.zeros(num_samples, 1)
        for i in range(num_samples):
            tbegin = i/(num_samples)
            tt = tbegin + torch.linspace(0,num_T,resolution)
            data[i] = torch.sin(2 * torch.pi * tt)
            labels[i] = tbegin;
        self.data = data.unsqueeze(1).unsqueeze(-1) # to be (B, 1, H, 1)
        self.labels = labels # to be (B,)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.data[idx], self.labels[idx]
        
    def transfer2ds(self, batch_size):
        X = self.data
        y = self.labels
        n_samples = X.shape[0]  # 获取第一个维度的长度
        num_batches = (n_samples + batch_size - 1) // batch_size  # 计算批次数量（向上取整）
        batches = []
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # 防止最后一个批次越界
            batch_X = X[start_idx:end_idx]  # 切片获取当前批次
            batch_y = y[start_idx:end_idx]
            batches.append([batch_X, batch_y])  # 存入字典，键为批次索引


        shuffled = batches.copy()
        random.shuffle(shuffled)
        splits = int(len(batches) * 0.8)
        train_ds = shuffled[0:splits]
        eval_ds =  shuffled[splits:]
        return (train_ds, eval_ds)

class CustomDataset_FreeFall(Dataset):
    def __init__(self, resolution=128):
        super().__init__()
        # ######################
        begin_g = 5.0; 
        self.scalar_g = end_g = 15.0; 
        num_g = 1000;
        
        self.scalar_m = m = 1.0
        self.scalar_T = T = 2.0
        
        num_samples = num_g
        self.num_cond = num_cond = 1
        # ######################
        data = torch.zeros(num_samples, 2 ,resolution) # q, p
        labels = torch.zeros(num_samples, num_cond)
        for i, gg in enumerate(np.linspace(begin_g, end_g, num_g)):
            tt = torch.linspace(0, T ,resolution)
            
            data[i, 0] = 0.5 * gg * tt**2 / (0.5 * self.scalar_g * self.scalar_T**2)
            data[i, 1] = m * gg * tt / (self.scalar_m * self.scalar_g * self.scalar_T)
            labels[i] = gg / (self.scalar_g)
        self.data = data.unsqueeze(-1) # to be (B, C, H, 1)
        self.labels = labels # to be (B,1)
        self.labels_distributed = None
    
    def IfReverseScalar(self):
        self.labels *= self.end_g
        self.data[:,0] *= (0.5 * self.scalar_g * self.scalar_T**2)
        self.data[:,1] *= (self.scalar_m * self.scalar_g * self.scalar_T)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.data[idx], self.labels[idx]
    
    def transfer2ds(self, batch_size):
        X = self.data
        y = self.labels
        y_dis = self.labels_distributed
        n_samples = X.shape[0]  # 获取第一个维度的长度
        num_batches = (n_samples + batch_size - 1) // batch_size  # 计算批次数量（向上取整）
        batches = []
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # 防止最后一个批次越界
            batch_X = X[start_idx:end_idx]  # 切片获取当前批次
            batch_y = y[start_idx:end_idx]
            batches.append([batch_X, batch_y])  # 存入字典，键为批次索引


        shuffled = batches.copy()
        random.shuffle(shuffled)
        splits = int(len(batches) * 0.8)
        train_ds = shuffled[0:splits]
        eval_ds =  shuffled[splits:]
        return (train_ds, eval_ds)
    
class CustomDataset_BubbleNV(Dataset):
    def __init__(self, resolution=128):
        super().__init__()
        # ######################
        self.scalar_m = m = 1.0
        self.num_T = num_T = 1
        
        begin_q = -1.; end_q = 1.0; 
        num_q = 20
        
        begin_p = -1.0; end_p = 1.0
        num_p = 20
        
        begin_k = 0.5; end_k = 1.5
        num_k = 20
    
        num_samples = num_q * num_p * num_k
        self.num_cond = num_cond = 3
        # ######################
        data = torch.zeros(num_samples, 2 ,resolution) # q, p
        labels = torch.zeros(num_samples, num_cond)
        idx = 0
        for qq0 in np.linspace(begin_q, end_q, num_q):
            for pp0 in np.linspace(begin_p, end_p, num_p):
                for kk in np.linspace(begin_k, end_k, num_k):
                    omega = np.sqrt(kk/m) 
                    tt = torch.linspace(0, 2.* np.pi * num_T/omega, resolution)     
                    A = qq0; B = pp0/omega/m       
                    data[idx, 0] = A * torch.cos(omega * tt) + B * torch.sin(omega * tt)
                    data[idx, 1] = m * omega * ( - A * torch.sin(omega * tt) + B * torch.cos(omega * tt)  )
                    labels[idx] = torch.tensor([qq0, pp0, kk])
                    idx += 1

        self.data = data.unsqueeze(-1) # to be (B, C, H, 1)
        self.labels = labels # to be (B,num_cond)
        self.labels_distributed = None
    
    def IfReverseScalar(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.data[idx], self.labels[idx]
    
    def transfer2ds(self, batch_size):
        X = self.data
        y = self.labels
        y_dis = self.labels_distributed
        n_samples = X.shape[0]  # 获取第一个维度的长度
        num_batches = (n_samples + batch_size - 1) // batch_size  # 计算批次数量（向上取整）
        batches = []
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # 防止最后一个批次越界
            batch_X = X[start_idx:end_idx]  # 切片获取当前批次
            batch_y = y[start_idx:end_idx]
            batches.append([batch_X, batch_y])  # 存入字典，键为批次索引


        shuffled = batches.copy()
        random.shuffle(shuffled)
        splits = int(len(batches) * 0.8)
        train_ds = shuffled[0:splits]
        eval_ds =  shuffled[splits:]
        return (train_ds, eval_ds)

class CustomDataset_BubbleV(Dataset):
    def __init__(self, resolution=1024):
        super().__init__()
        # ######################
        self.m = m = 1.0
        
        begin_q = 0.5 
        self.end_q = end_q = 2.0; 
        num_q = 20
        
        pp_fixed = 0.0
        
        begin_k = 0.5
        self.end_k = end_k = 1.5
        num_k = 20
        
        begin_mu = 1.5
        self.end_mu = end_mu = 3.0
        num_mu = 20
    
        num_samples = num_q * num_k * num_mu
        self.num_cond = num_cond = 3
        
        # ######################
        data = torch.zeros(num_samples, 2 ,resolution) # q, p
        labels = torch.zeros(num_samples, num_cond) #[q, k, mu]
        idx = 0
        tt = torch.linspace(0, 50., resolution)
        for qq0 in torch.linspace(begin_q, end_q, num_q):
            for kk in torch.linspace(begin_k, end_k, num_k):
                for mu in torch.linspace(begin_mu, end_mu, num_mu):
                        delta = (mu/m)**2 - 4 * kk / m
                        if delta > 0:
                            r1 = (- mu/m + torch.sqrt(delta)) / 2
                            r2 = (- mu/m - torch.sqrt(delta)) / 2
                            A = (pp_fixed / m - r2 * qq0) / (r1 - r2)
                            B = qq0 - A 
                            data[idx, 0]  = A * torch.exp(r1 * tt) + B * torch.exp(r2 * tt)
                            data[idx, 1] = ( A * r1 * torch.exp(r1 * tt) + B * r2 * torch.exp(r2 * tt) ) * m
                        elif delta == 0:
                            r = - mu / m / 2
                            A = qq0
                            B = pp_fixed/m - r * A
                            data[idx, 0] = (A + B * tt) * np.exp(r * tt)
                            data[idx, 1] = ( B * torch.exp(r * tt) + (A + B * tt) * r * torch.exp(r * tt) ) * m
                        else:
                            A = qq0
                            B = (pp_fixed/m + mu/(2 * m) * qq0) / torch.sqrt(kk/m - (mu/m)**2/4)   
                            a = - mu/(2 * m)
                            w = torch.sqrt(kk/m - (mu/m)**2/4)
                            data[idx, 0]  = torch.exp(a * tt) * ( A * torch.cos(w * tt) + B * torch.sin(w * tt) )
                            data[idx, 1]  = (  a * torch.exp(a * tt) * ( A * torch.cos(w * tt) + B * torch.sin(w * tt) ) + torch.exp(a * tt) * ( - A * w * torch.sin(w * tt) + B * w * torch.cos(w * tt) ) ) * m
                        labels[idx] = torch.tensor([qq0, kk, mu])
                        idx += 1

        self.data = data.unsqueeze(-1) # to be (B, C, H, 1)
        self.labels = labels # to be (B,num_cond)
        self.labels_distributed = None
        
    def IfReverseScalar(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.data[idx], self.labels[idx]
    
    def transfer2ds(self, batch_size):
        X = self.data
        y = self.labels
        y_dis = self.labels_distributed
        n_samples = X.shape[0]  # 获取第一个维度的长度
        num_batches = (n_samples + batch_size - 1) // batch_size  # 计算批次数量（向上取整）
        batches = []
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # 防止最后一个批次越界
            batch_X = X[start_idx:end_idx]  # 切片获取当前批次
            batch_y = y[start_idx:end_idx]
            batches.append([batch_X, batch_y])  # 存入字典，键为批次索引


        shuffled = batches.copy()
        random.shuffle(shuffled)
        splits = int(len(batches) * 0.8)
        train_ds = shuffled[0:splits]
        eval_ds =  shuffled[splits:]
        return (train_ds, eval_ds)
    
class CustomDataset_VortexSys(Dataset):
    def __init__(self, resolution=128):
        super().__init__()
        # ######################
        self.num_T = num_T = 1
        
        begin_q_2x = 0.5; end_q_2x = 2.0; 
        num_q = 50
        
        begin_gamma = -0.5; end_gamma = 2.0
        num_gamma = 40
        
    
        num_samples = num_q * num_gamma
        self.num_cond = num_cond = 2
        # ######################
        data = torch.zeros(num_samples, 4 ,resolution) # q, p
        labels = torch.zeros(num_samples, num_cond)
        idx = 0
        for q0_2x in np.linspace(begin_q_2x, end_q_2x, num_q):
            for gamma in np.linspace(begin_gamma, end_gamma, num_gamma):
                    xc = (0.0 * 1.0 + gamma * q0_2x)/ (1.0 + gamma + 1e-20)
                    omega = (gamma + 1.0) / 2. / np.pi / q0_2x**2
                    
                    t_end = 2.* np.pi * num_T/(omega+1e-10)
                    tt = torch.linspace(0, t_end, resolution)  
                    
                    A1 = np.abs(xc); A2 = np.abs(q0_2x - xc)
                    data[idx, 0] = xc * (1. - torch.cos(omega * tt))
                    data[idx, 1] = -xc * torch.sin(omega * tt) 
                    data[idx, 2] = xc + (q0_2x - xc) * torch.cos(omega * tt)
                    data[idx, 3] = + (q0_2x - xc) * torch.sin(omega * tt)
                    labels[idx] = torch.tensor([q0_2x, gamma])
                    idx += 1
            

        self.data = data.unsqueeze(-1) # to be (B, C, H, 1)
        self.labels = labels # to be (B,num_cond)
        self.labels_distributed = None
    
    def IfReverseScalar(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.data[idx], self.labels[idx]
    
    def transfer2ds(self, batch_size):
        X = self.data
        y = self.labels
        y_dis = self.labels_distributed
        n_samples = X.shape[0]  # 获取第一个维度的长度
        num_batches = (n_samples + batch_size - 1) // batch_size  # 计算批次数量（向上取整）
        batches = []
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # 防止最后一个批次越界
            batch_X = X[start_idx:end_idx]  # 切片获取当前批次
            batch_y = y[start_idx:end_idx]
            batches.append([batch_X, batch_y])  # 存入字典，键为批次索引


        shuffled = batches.copy()
        random.shuffle(shuffled)
        splits = int(len(batches) * 0.8)
        train_ds = shuffled[0:splits]
        eval_ds =  shuffled[splits:]
        return (train_ds, eval_ds)

class CustomDataset_Foil(Dataset):
    def __init__(self, height, width, IFall):
        super().__init__()
        # ######################
        if IFall:
            self.path = '../FoilCases/Foil_ABS_ALL'
        else:
            self.path = '../FoilCases/Foil_ABS'
            
        self.channel = channel = 3 # U V P 
        self.num_cond = num_cond = 4 # AOA, Creal, Cimag, Radius
        self.num_cond_distributed = num_cond_distributed = 2 # X, Y
        # ######################
        dirlist = os.listdir(self.path)
        flowlist = []
        for ff in dirlist:
            if ff.startswith('GridFlow_'):
                flowlist.append(ff[9:])
            
        num_samples = len(flowlist)
        
        flowdata = torch.zeros(num_samples, channel , height, width) # U, V, P
        
        labels = torch.zeros(num_samples, num_cond) # 
        labels_distributed = torch.zeros(num_samples, num_cond_distributed, height, width)
        idx = 0
        for flow in flowlist:
            
            filename = os.path.join(self.path, f'GridFlow_{flow}')
            X, Y , U, V, P = self.LoadZone(filename)
            flowdata[idx, 0, ...] = torch.from_numpy(U)
            flowdata[idx, 1, ...] = torch.from_numpy(V)
            flowdata[idx, 2, ...] = ( torch.from_numpy(P)) / (1.0 / 1.4 / (0.1)**2)
            labels_distributed[idx, 0, ...] = torch.from_numpy(X)
            labels_distributed[idx, 1, ...] = torch.from_numpy(Y)
            
            pattern = r'CrealNeg(\d+)_Cimag(\d+)_R(\d+)_Aoa(\d+)'
            match = re.search(pattern, flow)
    
            if match:
                creal = int(match.group(1)) * (- 1e-5)  # 第一个数字 (00000)
                cimag = int(match.group(2)) * (1e-5)     # 第二个数字 (00000)
                r_value = int(match.group(3)) * (1e-4)    # 第三个数字 (10200)
                aoa = int(match.group(4)) * (1e-1)        # 第四个数字 (030)
                print(f"loaded {idx+1} foils")
            else:
                raise ValueError(f"{flow} cannot be matched")
            labels[idx,0] = aoa # aoa
            labels[idx,1] = creal # creal
            labels[idx,2] = cimag  # cimag
            labels[idx,3] = r_value # radius
            
            idx += 1
            

        self.flowdata = flowdata # to be (B, C, H, W)
        self.labels = labels # to be (B,num_cond)
        self.labels_distributed = labels_distributed  # to be (B, 2, H, W)
    
    def IfReverseScalar(self):
        self.flowdata[:,2, ...] =  self.flowdata[:,2, ...]  * (1.0/1.4/ (0.1)**2)

    def IfScaleFoil2One(self):
        for idx in range(self.__len__()):
            _, creal, _, R = self.labels[idx,:]
            self.labels_distributed[idx, 0, ...] -= (creal - R) + 1./(creal - R)
            self.labels_distributed[idx, 0, ...] /= (2. * R - 2. * R/ (creal**2 - R**2))
            self.labels_distributed[idx, 1, ...] /= (2. * R - 2. * R/ (creal**2 - R**2))
    
    def __len__(self):
        return len(self.flowdata)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.flowdata[idx], self.labels[idx], self.labels_distributed[idx]
    
    def transfer2ds(self, batch_size):
        X = self.flowdata
        y = self.labels
        y_dis = self.labels_distributed
        n_samples = X.shape[0]  # 获取第一个维度的长度
        num_batches = (n_samples + batch_size - 1) // batch_size  # 计算批次数量（向上取整）
        batches = []
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # 防止最后一个批次越界
            batch_X = X[start_idx:end_idx]  # 切片获取当前批次
            batch_y = y[start_idx:end_idx]
            batch_y_dis = y_dis[start_idx:end_idx]
            batches.append([batch_X, batch_y, batch_y_dis])  # 存入字典，键为批次索引


        shuffled = batches.copy()
        random.shuffle(shuffled)
        splits = int(len(batches) * 0.8)
        train_ds = shuffled[0:splits]
        eval_ds =  shuffled[splits:]
        return (train_ds, eval_ds)
    
    def LoadZone(self, filename: str):
        loaded = np.load(filename, allow_pickle=True)

        X,Y = loaded['Grid_X'], loaded['Grid_Y']
        U,V,P = loaded['Flow_U'], loaded['Flow_V'], loaded['Flow_P']
        loaded.close()
        
        return X,Y,U,V,P
    
if __name__ == '__main__':  # Windows 必需的保护块
    # 创建数据集实例
    import matplotlib.pyplot as plt
    dataset = CustomDataset_Foil(96,128,IFall=True)
    
