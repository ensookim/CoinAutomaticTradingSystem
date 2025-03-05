import pyupbit
import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time
import datetime
import datetime
import pytz
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()


def fetch_ohlcv_period(ticker, start, end, interval):
    start = start.replace(second=0)
    end = end.replace(second=0)
    dfs = []
    current_end = start
    #print(current_end)
    i=0
    while True:
        df = pyupbit.get_ohlcv(ticker, interval=interval, to =current_end, count=200)
        print(df)
        if df is None or df.empty:
            print(f"No data available for {ticker} in the specified interval.")
            # 또는 다른 처리를 수행할 수 있음
        else:  
            if 'datetime' not in df.columns:
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'datetime'}, inplace=True)  # 'index' 열의 이름을 

            if(len(df) != 200 ):
                print(current_end,f"{i} 번째 csv에서 문제 발생")#749
                input()
                
            df.to_csv(f"KRW-BTC/{i}.csv")
            i+=1
            print(df)
    
            dfs.append(df)

            current_end = current_end + timedelta(minutes = 195)

            time.sleep(0.7)
            if(current_end>end ):
                full_df = pd.concat(dfs)
                full_df.reset_index(drop=True, inplace=True)

                print(full_df)
                print(type(full_df))
                full_df.to_csv("KRW-BTC.csv")
                return full_df
    if dfs:
        full_df = pd.concat(dfs)
        full_df.reset_index(inplace=True)
        return full_df
    else:
        return pd.DataFrame()

def data_preprocessing(data):
    # CSV 파일 로드
    window = 14
    # 이동 평균선 계산
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['MA50'] = data['close'].rolling(window=50).mean()
    data['MA200'] = data['close'].rolling(window=200).mean()

    # 볼린저 밴드 계산
    data['STD20'] = data['close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['MA20'] + (data['STD20'] * 2)
    data['Bollinger_Lower'] = data['MA20'] - (data['STD20'] * 2)

    data['price_change'] = data['close'] - data['close'].shift(1)
    # 상승폭과 하락폭 계산
    data['upward'] = data['price_change'].apply(lambda x: x if x > 0 else 0)
    data['downward'] = data['price_change'].apply(lambda x: -x if x < 0 else 0)
    # 상승폭과 하락폭의 평균 계산
    data['average_upward'] = data['upward'].rolling(window=window).mean()
    data['average_downward'] = data['downward'].rolling(window=window).mean()
    # 상대강도(RS) 계산
    data['RS'] = data['average_upward'] / data['average_downward']
    # RSI 계산
    data['RSI'] = 100 - (100 / (1 + data['RS']))
    data = data.dropna()
    data = data.drop(columns=['price_change'])
       
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    

    return data

def chekout_time(base_dir):
    total_df = []
    for i in range(0,len(os.listdir(base_dir))):
        csv_path = base_dir + f"/{i}.csv"
        print(csv_path)
        data = pd.read_csv(csv_path)
        total_df.append(data)
    result = pd.concat(total_df, ignore_index=True) 
    
    duplicates = result[result['datetime'].duplicated()]
    if not duplicates.empty:
        print("Duplicate values found, removing duplicates...")
        result.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
        result.drop(columns=['Unnamed: 0'], inplace=True)
    
    processing_data = data_preprocessing(result)
    processing_data['datetime'] = pd.to_datetime(processing_data['datetime'])    
    start_time = processing_data['datetime'].iloc[0]
    end_time = processing_data['datetime'].iloc[-1]
    current_time = start_time
    
    
    
    Null_cnt=0
    header = ['close', 'open', 'high','low','volume','value','MA20','MA50','MA200',
              'STD20','Bollinger_Upper','Bollinger_Lower','RSI','year','month','day','hour','minute','isNull']
    
    final_data = []
    
    for index, row in tqdm(processing_data.iterrows(),"정제 중"):
        ok_ = (row['datetime'] == current_time)
        
        if(ok_ == False):
    
            while (current_time < row['datetime']):
                final_data.append([row['close'],row['open'],row['high'],row['low'],row['volume'],row['value'],
                                row['MA20'],row['MA50'],row['MA200'],row['STD20'],row['Bollinger_Upper'],row['Bollinger_Lower'], row['RSI'],
                                row['year'],row['month'],row['day'],row['hour'],row['minute'],
                                0
                                ])
                Null_cnt+=1
                current_time += timedelta(minutes=1)  
        else:
            final_data.append([row['close'],row['open'],row['high'],row['low'],row['volume'],row['value'],
                               row['MA20'],row['MA50'],row['MA200'],row['STD20'],row['Bollinger_Upper'],row['Bollinger_Lower'], row['RSI'],
                               row['year'],row['month'],row['day'],row['hour'],row['minute'],
                               1
                               ])
        
        current_time += timedelta(minutes=1)  
 
    
    with open("PreProcessing_data.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 헤더 쓰기
        writer.writerows(final_data)  # 데이터 쓰기
  
    
    print(f"{Null_cnt} 개의 누락 값 발견")
    exit(0)
    return result





# UTC 시간으로부터 한국 시간으로 변환
current_time = datetime.datetime.now(pytz.utc)

one_year_ago = current_time - timedelta(days=365)

# 지정된 기간 설정
start = one_year_ago
end = current_time

base_dir = "KRW-BTC"
ticker = "KRW-BTC"
fetch_ohlcv_period(ticker, 
                   start, 
                   end, 
                   interval="minute1")

total_csv = chekout_time(base_dir)
print(total_csv.info())



data_preprocessing(
    "/Capston_Design/DataParser/KRW-BTC.csv"
    )

# 분 단위로 데이터를 가져와서 기존 데이터에 없는 데이터만 추가

