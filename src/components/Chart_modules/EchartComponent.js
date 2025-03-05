import React, { useEffect, useRef } from "react";
import * as echarts from 'echarts';

function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}
function calculateMA(dayCount, data) {
  const result = [];
  for (let i = 0, len = data[1].length; i < len; i++) {
    if (i < dayCount) {
      result.push('-');
      continue;
    }
    let sum = 0;
    for (let j = 0; j < dayCount; j++) {
      sum += data[1][i - j];
    }
    result.push((sum / dayCount).toFixed(2));
  }
  return result;
}

const EchartComponent = ({ data, aiData, orders }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;
    const chart = echarts.init(chartRef.current);
    chartInstance.current = chart;
    const lastDate = new Date(data[0][data[0].length - 1]);

    const aiDataWithDates = aiData.map((value, index) => {
      const newDate = new Date(lastDate.getTime());
      newDate.setMinutes(newDate.getMinutes() + (index + 1));
      return [formatDate(newDate), value];
    });

    // 매수/매도 시점 계산
    let buy_point = -50;
    let sell_point = -50;

    orders.forEach(order => {
      if (order.type === "buy") {
        buy_point = aiDataWithDates.find(item => item[0] === order.time);
      } else if (order.type === "sell") {
        sell_point = aiDataWithDates.find(item => item[0] === order.time);
      }
    });


    const transformed_chart_data = data[1].map((_, index) => {
      return [
        data[1][index], // 시가
        data[2][index], // 종가
        data[3][index], // 높이
        data[4][index]  // 낮이
      ];
    });

    // 차트 옵션 설정
    const option = {
      title: {
        text: ''
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['BTC-candlestick', 'MA5', 'MA10', 'MA20', 'MA30', 'Prediction']
      },
      grid: {
        left: '10%',
        right: '10%',
        bottom: '15%'
      },
      xAxis: {
        type: 'category',
        data: [...data[0], ...aiDataWithDates.map(item => item[0])],
        scale: true,
        boundaryGap: true,
        axisLine: { onZero: false },
        splitLine: { show: false },
        min: 'dataMin',
        max: 'dataMax'
      },
      yAxis: {
        scale: true,
        splitArea: {
          show: true
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 80,
          end: 100
        },
        {
          show: true,
          type: 'slider',
          top: '90%',
          start: 50,
          end: 100
        }
      ],
      series: [
        {
          name: 'BTC-candlestick',
          type: 'candlestick',
          data: transformed_chart_data,
          itemStyle: {
            color: '#ec0000',
            color0: '#00da3c',
            borderColor: '#8A0000',
            borderColor0: '#008F28'
          },
        },
        {
          name: 'MA5',
          type: 'line',
          data: calculateMA(5, data),
          smooth: true,
          lineStyle: { opacity: 0.5 }
        },
        {
          name: 'MA10',
          type: 'line',
          data: calculateMA(10, data),
          smooth: true,
          lineStyle: { opacity: 0.5 }
        },
        {
          name: 'MA20',
          type: 'line',
          data: calculateMA(20, data),
          smooth: true,
          lineStyle: { opacity: 0.5 }
        },
        {
          name: 'MA30',
          type: 'line',
          data: calculateMA(30, data),
          smooth: true,
          lineStyle: { opacity: 0.5 }
        },
        {
          name: 'Prediction',
          type: 'line',
          data: aiDataWithDates.map(item => [item[0], item[1]]),
          smooth: true,
          lineStyle: { color: '#ff00ff', width: 2 },
          itemStyle: { color: '#ff00ff' },
          markPoint: {
            data: [
              {
                name: 'Buy Point',
                coord: buy_point,
                value: 'Buy!',
                itemStyle: {
                  color: '#ff0000',
                  borderColor: '#ffffff',
                  borderWidth: 2,
                  borderType: 'solid'
                },
                label: {
                  show: true,
                  formatter: 'Buy!',
                  color: '#ffffff',
                  backgroundColor: '#ff0000'
                }
              },
              {
                name: 'Sell Point',
                coord: sell_point,
                value: 'Sell!',
                itemStyle: {
                  color: '#00ff00',
                  borderColor: '#ffffff',
                  borderWidth: 2,
                  borderType: 'solid'
                },
                label: {
                  show: true,
                  formatter: 'Sell!',
                  color: '#ffffff',
                  backgroundColor: '#00ff00'
                }
              }
            ]
          }
        }
      ]
    };

    chart.setOption(option);

    return () => {
      chart.dispose();
    };
  }, [data, aiData, orders]); // 데이터가 변경될 때만 다시 렌더링

  return <div ref={chartRef} style={{ width: '100%', height: '92%' }} />;
};

export default EchartComponent;
