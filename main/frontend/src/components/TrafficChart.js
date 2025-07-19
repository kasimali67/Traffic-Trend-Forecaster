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
      label: 'Traffic',
      data: [],
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }]
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await axios.get('http://localhost:8000/data');
        const trafficData = result.data;
        setData({
          labels: trafficData.map(d => new Date(d.date).toLocaleString()),
          datasets: [{ ...data.datasets, data: trafficData.map(d => d.traffic) }]
        });
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    fetchData();
  }, []);

  return ;
};

export default TrafficChart;
