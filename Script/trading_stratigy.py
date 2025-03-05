import torch
import pyupbit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
from getmodel_timeseries import my_get_model

class RealTime_Trading_strategy():
    def __init__(self):
        self.timestep = 20
        self.tickers = ["KRW-BTC"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        
    
    def inference(self):
        input_tensor = self.data_processing()
        model_path = "/Capston_Design/DataParser/result/moment_512_60_model/111.pt"
        
        model = self.load_model(model_path)
        
        pred = model(
            x_enc = input_tensor['past_values'].to(self.device).unsqueeze(1),
            input_mask = input_tensor['past_observed_mask'].to(self.device)
        )
        
        past_value = input_tensor['past_values'].squeeze(0)
        predict = pred.forecast.squeeze(0).squeeze(0)
        self.plot_predictions(past_value, predict)
        return past_value,predict
    
    def plot_predictions(self,past_value, predict):
        # Convert tensors to numpy arrays if necessary
        if hasattr(past_value, 'cpu'):
            past_value = past_value.detach().cpu().numpy()
        if hasattr(predict, 'cpu'):
            predict = predict.detach().cpu().numpy()

        # Create a new figure
        plt.figure(figsize=(12, 6))
        
        # Plot past values
        plt.plot(past_value, label='Past Values', color='blue')
        
        # Plot predicted values
        plt.plot(range(len(past_value), len(past_value) + len(predict)), predict, label='Predicted Values', color='red')
        
        # Adding labels and title
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Past Values and Predicted Values')
        plt.legend()
        
        # Show plot
        plt.savefig("test.png")

    def load_model(self,model_path):
        Getmodel_obj = my_get_model('moment')
        model = Getmodel_obj.get_model()
        model.load_state_dict(torch.load(model_path)) 
        return model    
    
    def data_processing(self):
        data_df = self.get_current_data()
        print(data_df)
        current_time = data_df['datetime'].iloc[0]
        result_data = []
        input_tensor = {}
        for index, row in tqdm(data_df.iterrows(),"정제 중"):
            ok_ = (row['datetime'] == current_time)
            if(ok_ == False):
                while (current_time < row['datetime']):
                    result_data.append([row['close'],0])
                    Null_cnt+=1
                    current_time += timedelta(minutes=1)  
            else:
                result_data.append([row['close'],1])
            current_time += timedelta(minutes=1)  
            
        np_result_data = np.array(result_data)
        input_tensor = {
            "past_values" : torch.tensor(np_result_data[:,0],dtype=torch.float).unsqueeze(0),
            "past_observed_mask" : torch.tensor(np_result_data[:,1],dtype=torch.float).unsqueeze(0)
        }
        return input_tensor      

    def get_current_data(self,interval = "minute1"):
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=512-199)
        
        
        time_window_size = 185
        total_ticker_data = []
        
        
        for ticker in self.tickers:
            
            while start_time < end_time:
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
                
        full_df = pd.concat(total_ticker_data)
        full_df.reset_index(drop=True, inplace=True) 
        full_df.drop_duplicates(subset='datetime', keep='last', inplace=True)
        
        return full_df
            
            #5 15 09 00 
            #5 17 09 00 
    
    def trading_Algorithm():
        return 0
    
    
system = RealTime_Trading_strategy()

while True:
    system.inference()
    time.sleep(60)


import pandas as pd
import pyupbit
from datetime import datetime, timedelta
import pytz
import time
