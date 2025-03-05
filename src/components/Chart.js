import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import {styled, createGlobalStyle } from 'styled-components';
import { io } from 'socket.io-client';
import PieChartComponent from './Chart_modules/PieChartComponent.js'
import EchartComponent from './Chart_modules/EchartComponent.js'
const socket = io('http://127.0.0.1:5000'); // Flask 서버 주소


// Styled Components
const GlobalStyle = createGlobalStyle`
  body, html {
    height: 100%; 
    background-color: #4ea685;
    margin: 0;
    padding: 0;
    overflow-y: auto; /* 페이지 스크롤 허용 */
  }
`;
const ChartContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  padding: 20px;
  overflow: hidden; /* 넘치는 부분 숨김 */

  @media (max-width: 768px) {
    padding: 10px;
  }
`;
const ChartWrapper = styled.div`
  width: 90vw;
  height: 70vh; /* 화면 높이의 70%에 50px을 더함 */
  background-color: #ffffff;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
  margin-bottom: 20px;
  margin-top: 20px;
  
  @media (max-width: 768px) {
    width: 100vw;
    height: calc(50vh + 30px); /* 모바일에서 화면 높이의 50%에 30px 추가 */
    padding: 10px;
  }
`;
const InfoBlock = styled.div`
  width: 90vw;
  background-color: #ffffff;
  padding: 20px;
  margin-top: 20px;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: space-around;
  align-items: center;

  @media (max-width: 768px) {
    width: 100vw;
    padding: 10px;
    flex-direction: column;
  }
`;
const BalanceTable = styled.table`
  width: 100%;
  border-collapse: collapse;

  th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
  }

  th {
    background-color: #4ea685;
    color: white;
  }

  td {
    background-color: #cccccc;
    color: #333333;
  }

  tr:hover {
    background-color: #d1e7dd;
  }

  @media (max-width: 768px) {
    th, td {
      padding: 5px;
    }
  }

  @media (max-width: 480px) {
    font-size: 12px;
  }
`;
const StyledButton = styled.button`
  background-color: #4ea685;
  color: white;
  font-size: 16px;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s;

  &:hover {
    background-color: #3a7e6b;
  }

  @media (max-width: 768px) {
    font-size: 14px;
    padding: 8px 16px;
  }
`;
const WelcomeMessage = styled.h1`
  margin-bottom: 20px;
  color: #fff;
  font-family: 'Arial', sans-serif;
  font-size: 64px;
  text-shadow: 3px 3px 3px rgba(5, 5, 5, 0.7);

  @media (max-width: 768px) {
    font-size: 48px;
  }

  @media (max-width: 480px) {
    font-size: 36px;
  }
`;


// Main Chart Component
function Main_Page() {
  const navigate = useNavigate();
  const location = useLocation();
  const [state, setState] = useState({
    chartData: [],
    KRW_balance: 0,
    expected_profit: 0,
    BTC_balance: 0,
    order_log: [],
    excuted_order: [],
    loading: true,
    error: null,
    username: location.state?.username || '',
    visibleOrders: 10,
  });

  useEffect(() => {
    fetchData();
    socket.on('connect', () => {
      console.log('Connected to server');
    });

    socket.on('data_update', (response) => {
      console.log('Received data from server:', response);
      if (response) {
        const { Inference_data, balances, chartData, order_log, excuted_order, expected_profit, profit_rate} = response;
        let KRW_balance = 0;
        let BTC_balance = 0;        
        balances.forEach(item => {
          item.forEach(real_val => {
            if (real_val.type === "KRW") {
              KRW_balance = Math.round(real_val.balance * 100) / 100;
            } else if (real_val.type === "BTC") {
              BTC_balance = real_val.balance;
            }
          });
        });
        setState(prevState => ({
          ...prevState,
          chartData: chartData,
          KRW_balance: KRW_balance,
          BTC_balance: BTC_balance,
          expected_profit: Math.round(expected_profit * 100) / 100,
          balance : balances,
          AI_Inference_data : Inference_data,
          order_log: order_log,
          excuted_order:excuted_order,
          profit_rate:profit_rate,
        }));

      }
      else{
        console.log("ASDf");
      }
    });
    
    return () => {
      socket.off('data_update');
    };
  }, []);

  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/initial_data');
      //동기식으로 Data 겟또다제 ㅋㅋ
      const { Inference_data, balances, chartData, order_log, excuted_order, expected_profit, profit_rate} = response.data;
      let KRW_balance = 0;
      let BTC_balance = 0;

      balances.forEach(item => {
        item.forEach(real_val => {
          if (real_val.type === "KRW") {
            KRW_balance = Math.round(real_val.balance * 100) / 100;
          } else if (real_val.type === "BTC") {
            BTC_balance = real_val.balance;
          }
        });
      });
      setState(prevState => ({
        ...prevState,
        chartData: chartData,
        KRW_balance: KRW_balance,
        BTC_balance: BTC_balance,
        expected_profit: Math.round(expected_profit * 100) / 100,
        balance : balances,
        AI_Inference_data : Inference_data,
        order_log: order_log,
        excuted_order:excuted_order,
        profit_rate:profit_rate,
        loading: false,
      }));

      console.log("Get Initial Data")
    } catch (error) {
      setState(prevState => ({ ...prevState, loading: false, error: 'Error fetching data' }));
      console.error('Error fetching data:', error);
    }
  };

  const handleBackToLogin = () => {
    navigate('/login');
  };

  if (state.loading) {
    return <div>Loading...</div>;
  }
  if (state.error) {
    return <div>{state.error}</div>;
  }

  return (
    <>
    <GlobalStyle />
    <ChartContainer>
      <WelcomeMessage>환영합니다 {state.username}님!</WelcomeMessage>
      <StyledButton onClick={handleBackToLogin}>로그인 돌아가기</StyledButton>
      <ChartWrapper>
        <BalanceTable>
              <th colSpan="3"><h1>BTC-Chart</h1></th>
        </BalanceTable>
          {state.chartData && state.chartData.length > 0 ? (
            <EchartComponent 
            data={state.chartData} 
            aiData={state.AI_Inference_data}//
            orders={state.excuted_order}
            style={{ width: '100%', height: '100%' }}/>) : (
            <div>No chart data available</div>
          )}
      </ChartWrapper>
      <InfoBlock>
      <PieChartComponent balance={state.balance} />
      <BalanceTable>
          <thead>
            <tr>
              <th colSpan="2"><h1>Current Profit</h1></th>
            </tr>
            <tr>
              <th style={{ fontSize: '20px' }}>Type</th>
              <th style={{ fontSize: '20px' }}>Value</th>
            </tr>
          </thead>
          <tbody style={{ fontFamily: 'Arial', fontSize: '20px' }}>
            <tr>
              <td>Quantity Held</td>
              <td>{state.profit_rate[0]}</td>
            </tr>
            <tr>
              <td>Total Purchase</td>
              <td>{state.profit_rate[1]}</td>
            </tr>
            <tr>
              <td>Total Evaluation</td>
              <td>{state.profit_rate[2]}</td>
            </tr>
            <tr>
              <td>Average Purchase Price</td>
              <td>{state.profit_rate[3]}</td>
            </tr>
            <tr>
              <td>Return on Investment (ROI)</td>
              <td style={{ color: state.profit_rate[4][0] > 0 ? 'red' : 'blue' }}>{state.profit_rate[4][0]}%</td>
            </tr>
            <tr>
              <td>Valuation Profit and Loss</td>
              <td style={{ color: state.profit_rate[4][1] > 0 ? 'red' : 'blue' }}>{state.profit_rate[4][1]}원</td>
            </tr>
          </tbody>
      </BalanceTable>
      </InfoBlock>
      <InfoBlock>
        <BalanceTable>
          <thead>
            <tr>
              <th colSpan="3"><h1>Account</h1></th>
            </tr>
            <tr>
              <th>KRW_balance</th>
              <th>BTC_balance</th>
              <th>expected_profit</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{state.KRW_balance}</td>
              <td>{state.BTC_balance}</td>
              <td>{state.expected_profit}</td>
            </tr>
          </tbody>
        </BalanceTable>
      </InfoBlock>
      <InfoBlock>
        <BalanceTable>
          <thead>
            <tr>
              <th colSpan="4"><h1>Order Log</h1></th>
            </tr>
            <tr>
              <th>market</th>
              <th>volume</th>
              <th>paid_fee</th>
              <th>time</th>
            </tr>
          </thead>
          <tbody>
            {state.order_log.length > 0 ? (
              state.order_log.slice(0, state.visibleOrders).map((order, index) => (
                <tr key={index}>
                  <td>{order.market}</td>
                  <td>{order.volume}</td>
                  <td>{order.paid_fee}</td>
                  <td>{order.time}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="4">No order log available.</td>
              </tr>
            )}
          </tbody>
          <tr>
            <td colSpan="4" style={{ textAlign: 'center', padding: '10px' }}>
              {state.order_log.length > state.visibleOrders && (
                <StyledButton onClick={() => setState(prevState => ({ 
                  ...prevState, 
                  visibleOrders: prevState.visibleOrders + 10 
                }))}>
                  이전 내역 불러오기
                </StyledButton>
              )}
            </td>
          </tr>
        </BalanceTable>

      </InfoBlock>
      
    </ChartContainer>
    </>
  );
}

export default Main_Page;