# AI Paper Trader

Cloud-ready educational paper-trading simulator. It uses market data and a transparent technical-analysis agent; it does **not** connect to a broker or place real orders.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload this folder to a Streamlit-compatible cloud deployment and set the main file to `app.py`.

## Features
- Market data via yfinance
- EMA20 / EMA50 / EMA200
- RSI, ATR and volume ratio
- BUY / SELL / HOLD agent score + confidence
- Manual paper trades
- Agent paper trades
- Virtual $10,000 portfolio
- Trade history
- Strategy backtest and drawdown
- Optional auto paper-trading loop

## Safety
This project is intentionally paper-only. It contains no broker credentials, no order API, and no real-money execution.
