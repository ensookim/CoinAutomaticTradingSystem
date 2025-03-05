from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import random
import pickle
import time

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
        
    

        self.mode = mode 
        
        self.save_npy_dir = "/Capston_Design/DataParser/Data/Processed_data" 
        
        # data = self.make_data_for_what()
 
        # total_length = len(data)
        # train_ratio = 0.9
        # test_ratio = 0.1
        # train_length = int(train_ratio * total_length)
        # test_length = int(test_ratio * total_length)
        
        # train_features = data[:train_length]
        # test_features = data[train_length:]
        # self.csv2pkl('train',train_features)
        # self.csv2pkl('test',test_features)   
        # exit(0)
        
    def csv2pkl(self,mode,features):
        os.makedirs(f'{self.save_npy_dir}/{mode}/label',exist_ok=True)
        os.makedirs(f'{self.save_npy_dir}/{mode}/features',exist_ok=True)
        
        for idx, feature in tqdm(enumerate(features)):
            past_values = feature['past_values'] 
            future_values = feature['future_values']
            past_observed_mask = feature['past_observed_mask'] 
            past_time_features = feature['past_time_features']
            
            past_observed_mask = past_observed_mask[:,np.newaxis]
            
            past_values_concat = np.concatenate((past_values,past_observed_mask),axis=1)
            past_values_concat = past_values_concat.transpose()
            
            past_values_concat = torch.tensor(past_values_concat, dtype=torch.float)
            future_values = torch.tensor(future_values,dtype=torch.float)
            with open(f'{self.save_npy_dir}/{mode}/features/{idx}.pkl', 'wb') as file:
                pickle.dump(past_values_concat, file)
            with open(f'{self.save_npy_dir}/{mode}/label/{idx}.pkl', 'wb') as file:
                pickle.dump(future_values, file)

    def __len__(self):
        return len(os.listdir(f"{self.save_npy_dir}/{self.mode}/features"))

    def __getitem__(self, idx):
        #start_time = time.time()
        with open(f'{self.save_npy_dir}/{self.mode}/features/{idx}.pkl', 'rb') as file:
            features = pickle.load(file)
            past_values, past_observed_mask = np.split(features, [9], axis=0)
        with open(f'{self.save_npy_dir}/{self.mode}/label/{idx}.pkl', 'rb') as file:
            future_values = pickle.load(file)
        return past_values,future_values,past_observed_mask.squeeze(0)

    def make_data_for_what(self):
#close,open,high,low,volume,value,MA20,MA50,MA200,STD20,Bollinger_Upper,Bollinger_Lower,RSI,year,month,day,hour,minute,isNull
        data = pd.read_csv(self.data_path) #1200 360    
        window_size = self.feature_length + self.label_length
        total_item_length = len(data)-window_size + 1
        
        total_data = []
        columns_to_drop = ['open','high','low','MA50']
        #volume,value,MA20,MA200,STD20,Bollinger_Upper,Bollinger_Lower,RSI
        
        data.drop(columns=columns_to_drop,inplace=True)
        
        total_values = data.loc[:,'close':'RSI'].to_numpy()
        total_future_values = data.loc[:,'close'].to_numpy()
        total_time_features = data.loc[:,'year':'minute'].to_numpy()

        total_observed_mask = data.loc[:,'isNull'].to_numpy()
        for index in tqdm(range(0,total_item_length),"데이터 가공"):
            data_dict = {
                'past_values': total_values[index:window_size-self.label_length+index],
                'future_values': total_future_values[index+self.feature_length:window_size+index],
                'past_time_features': total_time_features[index:window_size-self.label_length+index],
                'future_time_features': total_time_features[index+self.feature_length:window_size+index],
                'past_observed_mask': total_observed_mask[index:window_size-self.label_length+index],
                'future_observed_mask': total_observed_mask[index+self.feature_length:window_size+index]
            }
            total_data.append(data_dict)

        return total_data



def MV_Coin_Dataloader(dataset_dir,mode,batch_size):
    loader = Coin_Timeseries_Dataset(dataset_dir,mode)
    return DataLoader(loader,batch_size=batch_size,shuffle=False,pin_memory=True)
