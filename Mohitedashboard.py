import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------
# Setup
# ---------------------------------------------------
st.set_page_config(page_title="War Room Command Center", layout="wide")
st_autorefresh(interval=60 * 1000, key="refresh")

st.title("⚡ WAR ROOM – Multi Asset Command Center")

# ---------------------------------------------------
# Watchlist
# ---------------------------------------------------
watchlist = [
    "RELIANCE.NS",
    "BEL.NS",
    "HDFCBANK.NS",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "GC=F"  # Gold Futures
]

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ai_forecast(data):
    if len(data) < 30:
        return None, None
    
    data = data.dropna().copy()
    data['Date_O'] = pd.to_datetime(data.index).map(pd.Timestamp.toordinal)

    X = data[['Date_O']].values[-30:]
    y = data['Close'].values[-30:]

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[data['Date_O'].iloc[-1] + 1]])
    prediction = float(model.predict(next_day)[0])
    confidence = model.score(X, y)

    return prediction, confidence

# ---------------------------------------------------
# Main Panel Loop
# ---------------------------------------------------
rows = []

for ticker in watchlist:

    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)

        # Fix MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            continue

        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        data['RSI'] = calculate_rsi(data['Close'])

        price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        change_pct = ((price - prev_price) / prev_price) * 100
        rsi = float(data['RSI'].iloc[-1])
        ema = float(data['EMA200'].iloc[-1])

        forecast, confidence = ai_forecast(data)

        # Signal Logic
        if price > ema and rsi < 35:
            signal = "🔥 BUY"
        elif price < ema and rsi > 65:
            signal = "📉 SELL"
        else:
            signal = "⚖️ NEUTRAL"

        # Currency
        symbol = "₹" if ".NS" in ticker else "$"

        rows.append({
            "Ticker": ticker,
            "Price": f"{symbol}{price:,.2f}",
            "Change %": f"{change_pct:.2f}%",
            "RSI": f"{rsi:.1f}",
            "Trend vs EMA200": "Above" if price > ema else "Below",
            "AI Forecast": f"{symbol}{forecast:,.2f}" if forecast else "N/A",
            "Confidence": f"{confidence:.2f}" if confidence else "N/A",
            "Signal": signal
        })

    except Exception as e:
        st.warning(f"Error loading {ticker}")

# ---------------------------------------------------
# Display Panel
# ---------------------------------------------------
df = pd.DataFrame(rows)

st.dataframe(df, use_container_width=True)

st.caption("Auto-refresh every 60 seconds • AI Forecast = 30-day Linear Regression • For educational use only")