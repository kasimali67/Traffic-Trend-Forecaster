from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
import torch.nn as nn

app = FastAPI()

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_data():
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    traffic = pd.Series(range(100))
    df = pd.DataFrame({'date': dates, 'traffic': traffic})
    return df

class TrafficInput(BaseModel):
    past_traffic: List[float]

df = load_data()
model = LSTMForecaster()
model.eval()

@app.get('/')
def read_root():
    return {"message": "Traffic Trend Forecaster API is running."}

@app.get('/data')
def get_data():
    return df.tail(10).to_dict(orient='records')

@app.post('/predict')
def predict_traffic(input: TrafficInput):
    x = torch.tensor(input.past_traffic, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        pred = model(x)
    return {"prediction": pred.item()}
