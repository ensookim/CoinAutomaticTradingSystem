import torch
import pyupbit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
from getmodel_timeseries import my_get_model
import pytz

class Inference_system():
    def __init__(self):
        self.timestep = 20
        self.tickers = ["KRW-BTC"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        
        model_path = "/workspace/model/107_epoch_model.pt"
        self.model = self.load_model(model_path)
        
    def inference(self):
        chartData, past_observed_mask = self.data_processing()#side info에는 open low, high 이거 정보 넣기
        
        
        pred = self.model(
            x_enc = torch.tensor(chartData['close']).unsqueeze(0).unsqueeze(1).to(self.device),
            input_mask = past_observed_mask.to(self.device)
        )
        predict = pred.forecast.squeeze(0).squeeze(0).detach().cpu().numpy().tolist()
        return predict, chartData

    def load_model(self,model_path):
        Getmodel_obj = my_get_model('moment')
        model = Getmodel_obj.get_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=False)) 
        return model    
    
    def data_processing(self):
        data_df = self.get_current_data()
        current_time = data_df['datetime'].iloc[0]
        result_data = []
        chartData = {}
        Null_cnt = 0
        for index, row in data_df.iterrows():
            ok_ = (row['datetime'] == current_time)
            if(ok_ == False):
                while (current_time < row['datetime']):
                    result_data.append([row['close'],0,row['open'],row['high'],row['low']])
                    Null_cnt+=1
                    current_time += timedelta(minutes=1)  
            else:
                result_data.append([row['close'],1,row['open'],row['high'],row['low']])
            current_time += timedelta(minutes=1)  
            
        np_result_data = np.array(result_data)
        #[time, open，close，lowest，highest]
        time_data = [x.strftime('%Y-%m-%d %H:%M:%S') for x in data_df['datetime']]
        ################################################################################이거랑 데이터 불러오는거 손보기
        chartData = {
            "time" : time_data,
            "open" : np_result_data[-512:,2].tolist(),
            "close": np_result_data[-512:,0].tolist(),
            "low"  : np_result_data[-512:,4].tolist(),
            "high" : np_result_data[-512:,3].tolist()
        }
        
        return chartData, torch.tensor(np_result_data[-512:,1],dtype=torch.float).unsqueeze(0)

    def get_current_data(self,interval = "minute1"):
        server_timezone = pytz.utc
        end_time = datetime.now(server_timezone)
        start_time = end_time - timedelta(minutes=512-200)
        time_window_size = 185
        total_ticker_data = []
        for ticker in self.tickers:
            while start_time <= end_time:
                data = pyupbit.get_ohlcv(ticker, interval=interval, to =start_time, count=200)
                if 'datetime' not in data.columns:
                    data.reset_index(inplace=True)
                    data.rename(columns={'index': 'datetime'}, inplace=True)     
                total_ticker_data.append(data)
                time.sleep(0.7) 
                
                time_diff = end_time - start_time 
                min_time_diff = (time_diff.total_seconds() / 60)-1
                if(min(time_window_size,min_time_diff)== 0):
                    break
                start_time += timedelta(minutes=min(time_window_size+1,min_time_diff))

        current_data = pyupbit.get_ohlcv("KRW-BTC", interval= "minute1",to=end_time, count=1)
        current_data.reset_index(inplace=True)
        current_data.rename(columns={'index': 'datetime'}, inplace=True)          
        
                
        full_df = pd.concat(total_ticker_data)
        full_df = pd.concat([full_df, current_data], ignore_index=True)
        full_df.reset_index(drop=True, inplace=True) 
        full_df.drop_duplicates(subset='datetime', keep='last', inplace=True)
        return full_df
            
#             #5 15 09 00 
#             #5 17 09 00 
# server_timezone = pytz.utc
# current_time_utc = datetime.now(server_timezone)
# system = Inference_system()
# asdf = system.get_current_data()

# print(asdf)
