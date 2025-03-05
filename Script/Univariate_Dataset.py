from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torch
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import random
def set_seed(seed):
    # Python 시드 설정
    random.seed(seed)

    # Numpy 시드 설정
    np.random.seed(seed)

    # PyTorch 시드 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 알고리즘 시드 설정 (선택 사항)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class Coin_Timeseries_Dataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, data_dir, mode="train"):
        self.data_path = data_dir
        self.feature_length = 512
        self.label_length = 60
        
    
        self.data = self.make_data_for_what()
 
        total_length = len(self.data)
        train_ratio = 0.8
        valid_ratio = 0.1
        test_ratio = 0.1
        train_length = int(train_ratio * total_length)
        val_length = int(valid_ratio * total_length)
        test_length = int(test_ratio * total_length)

   
        # 데이터를 나누기
        train_data = self.data[:train_length]
        val_data = self.data[train_length:train_length + val_length]
        test_data = self.data[train_length + val_length:]
        

        print("train : " , len(train_data))
        print("tset : " , len(test_data))
        print("valid : " , len(val_data))

        

        if mode == "train":
            self.features = train_data
        elif mode == "test":
            self.features = test_data[::self.label_length]

        elif mode == "valid":
            self.features = val_data
        else:
            print(mode)
            raise Exception(f'{mode} is not invalid mode type')       
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        past_values = torch.tensor(feature['past_values'],dtype=torch.float)
        future_values = torch.tensor(feature['future_values'],dtype=torch.float)
        
        past_time_features = torch.tensor(feature['past_time_features'],dtype=torch.float)
        future_time_features = torch.tensor(feature['future_time_features'],dtype=torch.float)
           
        past_observed_mask = torch.tensor(feature['past_observed_mask'],dtype=torch.float)
        future_observed_mask =  torch.tensor(feature['future_observed_mask'],dtype=torch.float)
        
        # print(past_values.shape,past_values.dtype)
        # print(future_values.shape, future_values.dtype)
        
        # print(past_time_features.shape, past_time_features.dtype)
        # print(future_time_features.shape,future_time_features.dtype)
        
        # print(past_observed_mask.shape, past_observed_mask.dtype)
        # print(future_observed_mask.shape, future_observed_mask.dtype)
        # exit(0)
        data_dict = {
            'past_values': past_values,
            'future_values': future_values,
            'past_time_features': past_time_features,
            'future_time_features': future_time_features,
            'past_observed_mask': past_observed_mask,
            'future_observed_mask': future_observed_mask
        }      
        return data_dict

    def make_data_for_what(self):
#close,open,high,low,volume,value,MA20,MA50,MA200,STD20,Bollinger_Upper,Bollinger_Lower,RSI,year,month,day,hour,minute,isNull
        data = pd.read_csv(self.data_path) #1200 360    
        window_size = self.feature_length + self.label_length
        total_item_length = len(data)-window_size + 1
        
        total_data = []
        
        
        total_values = data.loc[:,'close'].to_numpy()
        total_time_features = data.loc[:,'year':'minute'].to_numpy()
        
        
        total_observed_mask = data.loc[:,'isNull'].to_numpy()
        for index in tqdm(range(0,total_item_length),"데이터 가공"):
            
            
            data_dict = {
                'past_values': total_values[index:window_size-self.label_length+index],
                'future_values': total_values[index+self.feature_length:window_size+index],
                'past_time_features': total_time_features[index:window_size-self.label_length+index],
                'future_time_features': total_time_features[index+self.feature_length:window_size+index],
                'past_observed_mask': total_observed_mask[index:window_size-self.label_length+index],
                'future_observed_mask': total_observed_mask[index+self.feature_length:window_size+index]
            }

            total_data.append(data_dict)

        return total_data




   
    def add_minutes_to_future_time(self, future_time_tensor, minutes_to_add):

        future_datetime = datetime(int(future_time_tensor[0]), 
                                   int(future_time_tensor[1]), 
                                   int(future_time_tensor[2]), 
                                   int(future_time_tensor[3]), 
                                   int(future_time_tensor[4]))
        
        updated_future_datetime = [ future_datetime + timedelta(minutes=i) for i in range(1,minutes_to_add+1)]

        
        # 다시 tensor로 변환합니다.
        updated_future_time_tensor = torch.tensor([[dt.year, dt.month, dt.day, dt.hour, dt.minute] for dt in updated_future_datetime])
        return updated_future_time_tensor
    

def MY_Coin_Dataloader(dataset_dir,mode,batch_size):
    loader = Coin_Timeseries_Dataset(dataset_dir,mode)
    return DataLoader(loader,batch_size=batch_size,shuffle=False)
