import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrafficChart = () => {
  const [data, setData] = useState({
    labels: [],
    datasets: [{
      label: 'Network Traffic',
      data: [],
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      tension: 0.1
    }]
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await axios.get('http://localhost:8000/data');
        const trafficData = result.data;
        setData({
          labels: trafficData.map(d => new Date(d.date).toLocaleTimeString()),
          datasets: [{
            ...data.datasets[0],
            data: trafficData.map(d => d.traffic)
          }]
        });
      } catch (error) {
        console.error('Error fetching traffic data:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    
      Real-time Network Traffic Monitoring
      
    
  );
};

export default TrafficChart;
