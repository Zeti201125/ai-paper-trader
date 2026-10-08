import os, time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='AI Paper Trader', page_icon='🤖', layout='wide')

START_CASH = 10_000.0
MAX_RISK_PCT = 1.0

# ---------- Data ----------
@st.cache_data(ttl=60)
def load_data(symbol: str, period: str, interval: str):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def add_indicators(df):
    x = df.copy()
    x['EMA20'] = x['Close'].ewm(span=20, adjust=False).mean()
    x['EMA50'] = x['Close'].ewm(span=50, adjust=False).mean()
    x['EMA200'] = x['Close'].ewm(span=200, adjust=False).mean()
    delta = x['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x['RSI'] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        x['High'] - x['Low'],
        (x['High'] - x['Close'].shift()).abs(),
        (x['Low'] - x['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    x['ATR'] = tr.rolling(14).mean()
    x['VOL20'] = x['Volume'].rolling(20).mean()
    x['VOL_RATIO'] = x['Volume'] / x['VOL20']
    x['RET5'] = x['Close'].pct_change(5)
    x['RET20'] = x['Close'].pct_change(20)
    return x.dropna()


def agent_decision(x):
    r = x.iloc[-1]
    score = 0
    reasons = []
    checks = []

    # Trend
    if r.Close > r.EMA20:
        score += 1; reasons.append('ár EMA20 felett')
        checks.append(('Trend', 'Bullish', 1))
    else:
        score -= 1; reasons.append('ár EMA20 alatt')
        checks.append(('Trend', 'Bearish', -1))

    if r.EMA20 > r.EMA50:
        score += 1; reasons.append('EMA20 > EMA50')
        checks.append(('Trend strength', 'Bullish', 1))
    else:
        score -= 1; reasons.append('EMA20 < EMA50')
        checks.append(('Trend strength', 'Bearish', -1))

    if r.Close > r.EMA200:
        score += 1; reasons.append('hosszú távú trend felfelé')
    else:
        score -= 1; reasons.append('hosszú távú trend lefelé')

    # Momentum
    if 45 <= r.RSI <= 65:
        score += 1; reasons.append(f'RSI egészséges momentum ({r.RSI:.1f})')
    elif r.RSI < 30:
        score += 1; reasons.append(f'RSI túladott zóna ({r.RSI:.1f})')
    elif r.RSI > 70:
        score -= 1; reasons.append(f'RSI túlvett zóna ({r.RSI:.1f})')
    else:
        reasons.append(f'RSI semleges ({r.RSI:.1f})')

    # Volume confirmation
    if r.VOL_RATIO > 1.2:
        score += 1 if r.RET5 > 0 else -1
        reasons.append(f'forgalom megerősítés ({r.VOL_RATIO:.1f}x)')

    if score >= 3:
        action = 'BUY'
    elif score <= -3:
        action = 'SELL'
    else:
        action = 'HOLD'

    confidence = min(95, 50 + abs(score) * 8)
    return action, score, confidence, reasons, checks


def reset_state():
    st.session_state.cash = START_CASH
    st.session_state.shares = 0.0
    st.session_state.trades = []
    st.session_state.equity = []
    st.session_state.last_auto_trade = None


if 'cash' not in st.session_state:
    reset_state()

# ---------- Sidebar ----------
with st.sidebar:
    st.header('⚙️ Agent settings')
    symbol = st.text_input('Ticker', 'SPY').upper().strip()
    period = st.selectbox('History', ['6mo', '1y', '2y', '5y'], index=1)
    interval = st.selectbox('Interval', ['1d', '1h'], index=0)
    auto = st.toggle('Auto paper-trading', False)
    risk = st.slider('Max risk / trade', 0.25, 1.0, 1.0, 0.25)
    if st.button('Reset simulator', use_container_width=True):
        reset_state(); st.rerun()
    st.divider()
    st.caption('Paper trading only. No real orders, broker connection or real-money account is used.')

# ---------- Load ----------
try:
    raw = load_data(symbol, period, interval)
except Exception as exc:
    st.error(f'Adatbetöltési hiba: {exc}')
    st.stop()

if raw.empty:
    st.error('Nem található adat ehhez a tickerhez.')
    st.stop()

x = add_indicators(raw)
if len(x) < 50:
    st.error('Túl kevés adat az elemzéshez.')
    st.stop()

price = float(x.Close.iloc[-1])
action, score, confidence, reasons, checks = agent_decision(x)
portfolio = st.session_state.cash + st.session_state.shares * price
pnl = portfolio - START_CASH

# ---------- Header ----------
st.title('🤖 AI Paper Trader')
st.caption(f'{symbol} · {interval} · utolsó frissítés: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric('Price', f'${price:,.2f}')
m2.metric('Agent', action)
m3.metric('Confidence', f'{confidence}%')
m4.metric('Portfolio', f'${portfolio:,.2f}', f'{pnl:+,.2f}')
m5.metric('RSI', f'{x.RSI.iloc[-1]:.1f}')

# ---------- Chart + Agent ----------
left, right = st.columns([2.1, 1])
with left:
    st.subheader('📈 Market')
    st.line_chart(x[['Close', 'EMA20', 'EMA50', 'EMA200']].tail(180), height=420)

with right:
    st.subheader('🧠 Agent decision')
    st.metric('Score', f'{score:+d}')
    for r in reasons:
        st.write('•', r)
    st.progress(confidence / 100)
    st.caption('A jel technikai szabályokból készül; nem garantálja a jövőbeli ármozgást.')

# ---------- Position ----------
st.divider()
p1, p2, p3 = st.columns(3)
p1.metric('Virtual cash', f'${st.session_state.cash:,.2f}')
p2.metric('Position', f'{st.session_state.shares:.4f} shares')
p3.metric('Position value', f'${st.session_state.shares * price:,.2f}')

qty = st.number_input('Manual paper quantity', min_value=0.0, value=1.0, step=1.0)
b1, b2, b3 = st.columns(3)
with b1:
    if st.button('🟢 BUY (paper)', use_container_width=True):
        cost = qty * price
        if qty <= 0 or cost > st.session_state.cash:
            st.warning('Nincs elég virtuális cash.')
        else:
            st.session_state.cash -= cost
            st.session_state.shares += qty
            st.session_state.trades.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'side':'BUY', 'qty':qty, 'price':price, 'reason':'manual'})
            st.rerun()
with b2:
    if st.button('🔴 SELL (paper)', use_container_width=True):
        if qty <= 0 or qty > st.session_state.shares:
            st.warning('Nincs elég virtuális pozíció.')
        else:
            st.session_state.cash += qty * price
            st.session_state.shares -= qty
            st.session_state.trades.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'side':'SELL', 'qty':qty, 'price':price, 'reason':'manual'})
            st.rerun()
with b3:
    if st.button('🤖 Run agent trade', use_container_width=True):
        # Conservative paper-only position sizing: cap at a small fraction of equity.
        budget = portfolio * (risk / 100)
        if action == 'BUY':
            q = min(budget / price, st.session_state.cash / price)
            if q > 0:
                st.session_state.cash -= q * price
                st.session_state.shares += q
                st.session_state.trades.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'side':'BUY', 'qty':q, 'price':price, 'reason':f'agent score {score}'})
        elif action == 'SELL' and st.session_state.shares > 0:
            q = min(st.session_state.shares, budget / price)
            st.session_state.cash += q * price
            st.session_state.shares -= q
            st.session_state.trades.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'side':'SELL', 'qty':q, 'price':price, 'reason':f'agent score {score}'})
        st.session_state.last_auto_trade = datetime.now().strftime('%H:%M:%S')
        st.rerun()

# ---------- Backtest ----------
st.divider()
st.subheader('🧪 Strategy backtest')
if st.button('Run backtest', use_container_width=True):
    bt = x.copy()
    bt['score'] = 0
    bt.loc[bt.Close > bt.EMA20, 'score'] += 1
    bt.loc[bt.Close < bt.EMA20, 'score'] -= 1
    bt.loc[bt.EMA20 > bt.EMA50, 'score'] += 1
    bt.loc[bt.EMA20 < bt.EMA50, 'score'] -= 1
    bt.loc[bt.Close > bt.EMA200, 'score'] += 1
    bt.loc[bt.Close < bt.EMA200, 'score'] -= 1
    bt.loc[(bt.RSI >= 45) & (bt.RSI <= 65), 'score'] += 1
    bt.loc[bt.RSI < 30, 'score'] += 1
    bt.loc[bt.RSI > 70, 'score'] -= 1
    bt['signal'] = np.where(bt.score >= 3, 1, np.where(bt.score <= -3, -1, 0))
    bt['position'] = bt.signal.replace(0, np.nan).ffill().fillna(0)
    bt['strategy_ret'] = bt.position.shift(1) * bt.Close.pct_change()
    bt['curve'] = (1 + bt.strategy_ret.fillna(0)).cumprod() * 10000
    total = bt.curve.iloc[-1] / 10000 - 1
    buyhold = bt.Close.iloc[-1] / bt.Close.iloc[0] - 1
    dd = (bt.curve / bt.curve.cummax() - 1).min()
    c1,c2,c3 = st.columns(3)
    c1.metric('Strategy return', f'{total*100:+.2f}%')
    c2.metric('Buy & hold', f'{buyhold*100:+.2f}%')
    c3.metric('Max drawdown', f'{dd*100:.2f}%')
    st.line_chart(bt[['curve']], height=300)

# ---------- History ----------
st.divider()
st.subheader('📋 Paper trade history')
if st.session_state.trades:
    st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True, hide_index=True)
else:
    st.caption('Még nincs paper trade.')

if auto:
    st.info('Auto paper-trading mód: 60 másodpercenként újraértékeli a stratégiát. Ez továbbra is csak szimuláció.')
    time.sleep(60)
    st.rerun()
