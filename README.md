# Traffic-Trend-Forecaster


An LSTM-based time-series model to predict next-hour network traffic with automated data processing pipeline and real-time visualization dashboard.

## Features

- LSTM PyTorch model for traffic forecasting (18% MAPE reduction)
- Automated data ingestion and preprocessing with Pandas
- 24-hour rolling window model retraining under 2 minutes
- FastAPI backend with REST endpoints
- React.js frontend with Chart.js visualization
- Real-time monitoring dashboard

## Setup Instructions

### Backend (Python/FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend (React.js)

```bash
cd frontend
npm install
npm start
```

## Technologies Used

- **Backend**: Python, FastAPI, PyTorch, Pandas, Matplotlib
- **Frontend**: React.js, Chart.js, Axios
- **ML**: LSTM neural networks for time-series forecasting
