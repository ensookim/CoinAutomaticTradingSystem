import os
import pytz
import time
import pyupbit
import warnings
import numpy as np
from datetime import datetime, timedelta
from Inference_system import Inference_system
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils._pytree")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 기본: '0', 1: INFO, 2: WARNING, 3: ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#이거 주문하는 방법
#그 min_max 이거 해서 매수 매도 날짜를 잡음.
#그 매수 시기가 됐을 때 내가 예측했던 값 += ??% 범주 안에 들어와있으면 구매
#매도는 날짜가 딱 됐을 떄 그냥 하기

output_file = 'output.csv'
import csv
class PositionManager:
    #얘가 해야하는 것
    #Prediction Data를 가지고 오면 buy, sell, 이거 포지션 잡고 결과(order log 넣어주기)
    def __init__(self, upbit, market, risk_per_trade=0.02):
        self.upbit = upbit
        self.market = market
        self.risk_per_trade = risk_per_trade
        self.position = None
        self.entry_price = None
        self.KRW_balance = 0

        
        
        

class TradingBot:
    def __init__(self, access, secret, market='KRW-BTC'):
        self.market = market
        self.inference_system = Inference_system()
        self.upbit = pyupbit.Upbit(access, secret)  
        self.fee_rate = 0.0005 
        self.balances = []
        self.excuted_orders = []
    def predict(self):
        predicted, chartData = self.inference_system.inference()
        return predicted, chartData
    def decide_action(self, current_price, predicted_prices):
        result = self.find_min_max(predicted_prices)
        balance = self.upbit.get_balances()
        self.balances = [] 
        for balance_ in balance:
            self.balances.append([{
                   'type' : balance_['currency'],
                    'balance' : balance_['balance']
                }])
            if balance_['currency'] =="KRW":
                self.KRW_balance = float(balance_['balance'])
            
        min_price = result['min_value']
        max_price = result['max_value']
        min_index = result['min_index']
        max_index = result['max_index']
        
        expect_buy_fee = min_price * self.fee_rate
        expect_sell_fee = max_price * self.fee_rate
        expected_buy_price = min_price + expect_buy_fee
        expected_sell_price = max_price - expect_sell_fee
        expected_profit = (expected_sell_price - expected_buy_price) * (self.KRW_balance / expected_buy_price)

        if expected_profit > 0 and self.excuted_orders == []:
            server_timezone = pytz.utc
            kr_time = datetime.now(server_timezone)
            kr_timezone = pytz.timezone('Asia/Seoul')
            buy_time = kr_time.astimezone(kr_timezone).replace(second=0, microsecond=0) + timedelta(minutes=min_index-1)
            sell_time = kr_time.astimezone(kr_timezone).replace(second=0, microsecond=0) + timedelta(minutes=max_index-1)
            self.place_order("buy", min_price, buy_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.place_order("sell",max_price, sell_time.strftime('%Y-%m-%d %H:%M:%S')) 
            print(f"expected profit : {expected_profit}")
        return expected_profit
    def find_min_max(self,arr):
        if len(arr) < 2:
            return None
        result_arr = []
        indexed_list = [(value, index) for index, value in enumerate(arr)]
        ascending_sorted = sorted(indexed_list, key=lambda x: x[0])
        descending_sorted = sorted(indexed_list, key=lambda x: x[0], reverse=True)
        for min_value, min_index in ascending_sorted:
            for max_value, max_index in descending_sorted:
                if min_value<max_value and min_index < max_index:
                    result_arr.append({"min_value":min_value,"min_index":min_index,"max_value":max_value,"max_index":max_index})
        return result_arr[0]
    def execute_trade(self, current_price):
        server_timezone = pytz.utc
        current_time_utc = datetime.now(server_timezone)
        
        kr_timezone = pytz.timezone('Asia/Seoul')
        current_time_kr = datetime.now(kr_timezone).replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
       
        for order in self.excuted_orders:
            
            print(order['type'], order['price'],order['time'])
            order_type = order['type']
            if order_type == "buy":
                if order['time'] == current_time_kr:
                    if abs(order['price']-current_price) < 1500000:
                        self.open_position(current_price)
                        self.excuted_orders.remove(order)
                    else:
                        print(f"예상 매수가와 실 가격의 차이가 있기에 주문 취소함. 차이 : {abs(order['price']-current_price)}")
                        self.excuted_orders = []
            elif order_type == "sell":
                if order['time'] == current_time_kr:
                    self.close_position(current_price)
                    self.excuted_orders.remove(order)                     
    def place_order(self, order_type, price, order_time):
        self.excuted_orders.append({
            "type": order_type,
            "price": price,
            "time": order_time
        })
        print(f"Placed {order_type} order at {price} for {order_time}")         
    def open_position(self,current_price):
        available_balance = self.upbit.get_balance("KRW")
        buy_amount = available_balance  # 5000원이 넘지 않도록 설정
        if buy_amount < 10000:
            print("잔액이 부족하여 매수를 실행할 수 없습니다.")
            return
        buy_amount = 10000
        self.upbit.buy_market_order(self.market, buy_amount)
        print(f"매수 실행: 가격 {current_price}, 주문 금액: {buy_amount}")
    def close_position(self,current_price):
        sell_amount = self.upbit.get_balance(self.market)
        self.upbit.sell_market_order(self.market, sell_amount)
        print(f"매도 실행: 가격 {current_price}, 매도 수량: {sell_amount}")       
    def get_order_log(self):
        order_list = []
        trades = self.upbit.get_order("KRW-BTC", state="done")
        for trade in trades:
            dt = datetime.fromisoformat(trade['created_at'])
            formatted_date = dt.strftime('%Y-%m-%d %H:%M:00')
            order_list.append({
                "market" : trade['market'],
                "volume" : trade['volume'],
                "paid_fee" : trade['paid_fee'],
                "time" : formatted_date,
            })
        return order_list
    def get_profit_rate(self, current_price):
        ##################################profit_rate#####################
        #"volume": ,
        #"total_KRW": ,
        #"total_investment_KRW":,
        #"avg_buy_price": ,
        #"profit_rate_won": 
        ##################################################################
        profit_rate = []
        balances = self.upbit.get_balances()
        for balance in balances:
            if balance['currency']=="BTC":
                volume = balance['balance']
                total_KRW = current_price * float(balance['balance'])
                total_investment_KRW = float(balance['avg_buy_price']) * float(balance['balance'])
                avg_buy_price = balance['avg_buy_price']
                profit_rate_won = [
                        round((current_price * float(balance['balance'])-(float(balance['avg_buy_price']) * float(balance['balance'])))/(float(balance['avg_buy_price']) * float(balance['balance']))*100,4),
                        round(current_price * float(balance['balance'])-(float(balance['avg_buy_price']) * float(balance['balance'])),4)
                    ]
                profit_rate = [volume, total_KRW, total_investment_KRW, avg_buy_price, profit_rate_won]
        if profit_rate == []:
            profit_rate = [0,0,0,0,[0,0]]
        return profit_rate
    def run(self):
        server_timezone = pytz.utc
        current_time_utc = datetime.now(server_timezone)
        current_price = pyupbit.get_ohlcv("KRW-BTC", interval= "minute1",to=current_time_utc, count=1)['close'].to_list()[0]
        predicted_prices, chartData = self.predict()
        expected_profit = self.decide_action(current_price, predicted_prices)
        self.execute_trade(current_price)
        
        
        
        order_list = self.get_order_log()
        profit_rate = self.get_profit_rate(chartData['close'][-1])
        chartData = [chartData['time'],chartData['open'],chartData['close'],chartData['low'],chartData['high']]  
        return chartData, expected_profit, self.excuted_orders, order_list, self.balances, predicted_prices, profit_rate

        
#initlal_list
#chartData, expected_profit, self.excuted_orders, order_list, self.balances, predicted_prices, profit_rate
#update####################################
#chartData - 1분단위[open, close, high, low]
#prediction data - 전체
#excuted orders - 전체
#order list - 이건 업데이트 될 때마다 가는게 좋을것 같은데
#balances - 전체
#profit_Rate - 전체


# if __name__ == "__main__":
#     access = "Rx87hxCR11Z9ILaffVQcUo4LzQoOxA1Ov4Q0sk3F" #본인의 Access Key
#     secret = "rCPHdcvNUOcBwOzjt3YfeMXu9kOd9bSlZX930rBk" #본인의 Secret Key
#     bot = TradingBot(access, secret)
#     result = bot.run("continuous")
#     print(result[0])
#     exit(0)

# 개발 남은거

# 1. 개같은 시뮹레이션 조지기
# 2. 로그인 할 떄 Upbit Access Key 받는 거 물어보기
# 3. 매수 매도 전략 변경
# -- prediction EMA내서 차트 업데이트 + 그 값 가지고 min, max 구해서 다시 매수, 매도 결정
# -- Position Manager 이거 따로 떼서 Hold, Buy, sell 결정
# 4. 새로운 모델 적용 D-PAD
# 5. 새로운 모델 적용 - 다변량 모델 학습(open, close, low, high)