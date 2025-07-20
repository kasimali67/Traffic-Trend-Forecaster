# Traffic Trend Forecaster

An LSTM-based time-series model to predict next-hour network traffic with automated data processing pipeline and real-time visualization dashboard.

## Features

- **LSTM PyTorch Model**: Reduces mean absolute percentage error by 18%
- **Automated Data Pipeline**: Pandas-based ingestion and preprocessing
- **Fast Model Retraining**: 24-hour rolling window retraining under 2 minutes
- **REST API**: FastAPI backend with prediction endpoints
- **Real-time Dashboard**: React.js frontend with Chart.js visualization
- **Live Monitoring**: WebSocket-based real-time traffic monitoring

## Technologies Used

- **Frontend**: React.js, TypeScript, Vite, TailwindCSS
- **Visualization**: Chart.js, React-Chart.js-2
- **Backend**: Python, FastAPI, PyTorch, Pandas
- **ML**: LSTM neural networks for time-series forecasting
- **Build Tools**: Vite, PostCSS, ESLint

## Setup Instructions

### Install Dependencies
```
npm install
```

### Development Server
```
npm run dev
```

### Build for Production
```
npm run build
```
