import React, { useEffect, useRef } from "react";
import axios from 'axios';
import * as echarts from 'echarts';

let bitcoinPrice = 0;
const fetchBitcoinPrice = () => {
  axios.get('https://api.upbit.com/v1/ticker?markets=KRW-BTC')
    .then(response => {
      bitcoinPrice = response.data[0].trade_price;
    })
    .catch(error => {
      console.error('에러 발생:', error);
    });
};

const PieChartComponent = ({ balance }) => {
    const chartRef = useRef(null);
  
    useEffect(() => {
      const chart = echarts.init(chartRef.current);
  
      fetchBitcoinPrice()
      const data = balance.map(item => {
        const balanceObj = item[0]; // 첫 번째 요소를 가져옵니다.
  
  
        if(balanceObj.type==="BTC"){
          balanceObj.balance = Number(balanceObj.balance)*bitcoinPrice;
        }
        return {
          value: Number(balanceObj.balance), // balance를 숫자로 변환합니다.
          name: balanceObj.type // type을 name으로 설정합니다.
        };
      });
      const option = {
        title: {
          text: 'Current Balance',
          left: 'center', // 제목을 중앙에 배치
          top: '5%', // 필요에 따라 상단 여백 조절
          textStyle: {
            fontSize: 24, // 글씨 크기 설정 (단위: px)
            fontFamily: 'Arial', // 폰트 패밀리 설정
            fontWeight: 'bold', // 글씨 굵기 설정 (예: 'normal', 'bold')
            color: '#333', // 글씨 색상 설정
          },
        },
        tooltip: {
          trigger: 'item'
        },
        legend: {
          bottom: '5%',
          left: 'center'
        },
        series: [
          {
            name: 'Access From',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2
            },
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: 40,
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data : data
          }
        ]
      };
      chart.setOption(option);
      return () => {
        chart.dispose();
      };
    }, [balance]);
  
    return (
        <div
        ref={chartRef}
        style={{ width: '50%', height: '400px' }}
        />
    );
};
export default PieChartComponent;