from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
import time
import threading
import os

from Auto_Trading_v3_real_time_update import TradingBot




initial_data = {
    "chartData" : [],
    "expected_profit" : 0,
    "order_log" : [],
    "excuted_order" : [],
    "balances": [],
    "profit_rate": [],
    "Inference_data" : []
}
continuous_data = {
    "chartData" : [],
    "expected_profit" : 0,
    "order_log" : [],
    "excuted_order" : [],
    "balances": [],
    "profit_rate": [],
    "Inference_data" : []
}
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, supports_credentials=True)  # 모든 출처 허용



access = "Rx87hxCR11Z9ILaffVQcUo4LzQoOxA1Ov4Q0sk3F" #본인의 Access Key
secret = "rCPHdcvNUOcBwOzjt3YfeMXu9kOd9bSlZX930rBk" #본인의 Secret Key
bot = TradingBot(access, secret)
# 데이터 파일 경로
USER_DATA_FILE = 'users.txt'



def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return []
    with open(USER_DATA_FILE, 'r') as file:
        users = [line.strip().split(',') for line in file]
    return [{"username": user[0], "password": user[1]} for user in users]

def save_user(username, password):
    with open(USER_DATA_FILE, 'a') as file:
        file.write(f"{username},{password}\n")

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    users = load_users()


    if any(user['username'] == data['username'] for user in users):
        return jsonify({"message": "Username already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    save_user(data['username'], hashed_password)
    print(data)
    return jsonify({"message": "User created successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    users = load_users()
    
    user = next((user for user in users if user['username'] == data['username']), None)
    print(user,bcrypt.check_password_hash(user['password'], data['password']))
    if bcrypt.check_password_hash(user['password'], data['password']):
        session['user'] = user['username']
        #print("tlqkf?")
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 401


@app.route('/initial_data', methods=['GET'])
def initial_():
    chartData, expected_profit, excuted_order, orders, balances, predicted_prices, profit_rate  = bot.run()
    
    initial_data['chartData'] = chartData
    initial_data['expected_profit'] = expected_profit
    initial_data['excuted_order'] = excuted_order
    initial_data['order_log'] = orders
    initial_data['balances'] = balances
    initial_data['profit_rate'] = profit_rate
    initial_data["Inference_data"] = predicted_prices        
    return jsonify(initial_data)

def update_data():
    previous_minute = datetime.now().minute
    while True:
        current_minute = datetime.now().minute
        time.sleep(1)
        if current_minute != previous_minute:
            print(current_minute)
            print("update_data 호출됨")
            chartData, expected_profit, excuted_order, orders, balances, predicted_prices, profit_rate  = bot.run()
            continuous_data['chartData'] = chartData
            continuous_data['expected_profit'] = expected_profit
            continuous_data['excuted_order'] = excuted_order
            continuous_data['order_log'] = orders
            continuous_data['balances'] = balances
            continuous_data['profit_rate'] = profit_rate
            print(profit_rate)
            continuous_data["Inference_data"] = predicted_prices   
            socketio.emit('data_update', continuous_data)
            previous_minute = current_minute
        

if __name__ == '__main__':
    threading.Thread(target=update_data, daemon=True).start()
    socketio.run(app, port=5000, debug=True)
    
    
    
    
    
    
# 개발 남은거########################################################
# 1. 클라이언트에 보내는 데이터 initial페이지로 한 번 update 페이지로 지속적으로 보내는거(업데이트 되는 데이터만 하나씩 
# 2. -------------------------------client 코드 정리(차트 모듈화 및 CSS 각자각자 하는 거로)
# 3. 새로운 모델 적용 D-PAD
# 4. 새로운 모델 적용 - 다변량 모델 학습(open, close, low, high)
# 5. 시1발롬의 시뮬레이션(차트 특정 시간 클릭 시 현재까지 프로그레스 바 -> series.areaStyle: {})
# 6. Access, secret key login 창에서 입력받고 개인으로 넘어갈 수 있게 만들기


