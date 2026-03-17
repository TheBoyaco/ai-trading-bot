import os, re, json, glob
from datetime import datetime, timezone

import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

# ================= CONFIG =================
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"     # 1m, 3m, 5m, 15m, 1h...
DEFAULT_LIMIT = 200         # velas
LOG_FILE = os.path.join(os.path.dirname(__file__), "paper_task.log")
SNAP_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
RISK_LOCK = os.path.join(os.path.dirname(__file__), "risk_lock.json")
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
# ==========================================

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    r = requests.get(BINANCE_KLINES, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open_time","open","high","low","close","volume"]]

def parse_log_trades(log_path: str):
    if not os.path.exists(log_path):
        return []

    side_re = re.compile(r"\b(BUY|SELL|COMPRA|VENTA)\b", re.IGNORECASE)
    price_re = re.compile(r"(price|precio)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")

    trades = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = side_re.search(line)
            if not m:
                continue

            side = m.group(1).upper()
            side = "BUY" if side in ("BUY","COMPRA") else "SELL"

            pm = price_re.search(line)
            price = float(pm.group(2)) if pm else None

            tm = ts_re.search(line)
            ts = None
            if tm:
                s = tm.group(1).replace(" ", "T")
                try:
                    ts = datetime.fromisoformat(s)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    else:
                        ts = ts.astimezone(timezone.utc)
                except Exception:
                    ts = None

            trades.append({"side": side, "price": price, "ts": ts, "raw": line.strip()})

    return trades[-200:]

def load_equity_from_snapshots(snap_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(snap_dir, "paper_state_*.json")))
    if not files:
        return pd.DataFrame(columns=["ts","capital"])
    rows = []
    for fp in files:
        name = os.path.basename(fp).replace("paper_state_", "").replace(".json", "")
        try:
            ts = datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        try:
            with open(fp, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            cap = float(d.get("capital", 0.0))
        except Exception:
            continue
        rows.append((ts, cap))
    return pd.DataFrame(rows, columns=["ts","capital"])

def load_risk_lock(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None

def add_trade_markers(fig, df, trades):
    if df.empty or not trades:
        return fig

    times = df["open_time"]
    buy_x, buy_y, sell_x, sell_y = [], [], [], []

    for t in trades:
        side = t.get("side")
        price = t.get("price")
        ts = t.get("ts")

        if ts is not None:
            try:
                idx = (times - pd.Timestamp(ts)).abs().idxmin()
                x = df.loc[idx, "open_time"]
                y = price if price is not None else df.loc[idx, "close"]
            except Exception:
                x = df.iloc[-1]["open_time"]
                y = price if price is not None else df.iloc[-1]["close"]
        else:
            x = df.iloc[-1]["open_time"]
            y = price if price is not None else df.iloc[-1]["close"]

        if side == "BUY":
            buy_x.append(x); buy_y.append(y)
        elif side == "SELL":
            sell_x.append(x); sell_y.append(y)

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers", name="BUY",
            marker_symbol="triangle-up", marker_size=12
        ))
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers", name="SELL",
            marker_symbol="triangle-down", marker_size=12
        ))

    return fig

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Live Bot Viewer", layout="wide")
st.title("📡 Live Trading Viewer (sin tocar tu bot)")

symbol = st.sidebar.text_input("Symbol", DEFAULT_SYMBOL).upper().strip()
interval = st.sidebar.selectbox("Interval", ["1m","3m","5m","15m","30m","1h","4h"], index=0)
limit = st.sidebar.slider("Candles", 50, 500, DEFAULT_LIMIT, 10)

colA, colB, colC = st.columns([2,1,1])

risk = load_risk_lock(RISK_LOCK)
with colB:
    st.subheader("🛑 Risk lock")
    if risk is None:
        st.info("No existe risk_lock.json aún.")
    else:
        st.json(risk)

with colC:
    st.subheader("📈 Equity (snapshots)")
    eq = load_equity_from_snapshots(SNAP_DIR)
    if eq.empty:
        st.warning("No hay snapshots aún. Ejecuta snapshot_state.py varias veces.")
    else:
        st.line_chart(eq.set_index("ts")["capital"])

with colA:
    st.subheader("📊 Velas + BUY/SELL (desde paper_task.log)")
    try:
        df = fetch_klines(symbol, interval, limit)
        fig = go.Figure(data=[go.Candlestick(
            x=df["open_time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=f"{symbol} {interval}"
        )])

        trades = parse_log_trades(LOG_FILE)
        fig = add_trade_markers(fig, df, trades)
        fig.update_layout(height=650, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Log leído: {LOG_FILE} | Señales detectadas: {len(trades)}")
        if trades:
            st.write("Últimas señales (log):")
            st.dataframe(pd.DataFrame(trades[-15:])[["side","price","ts","raw"]], use_container_width=True)
    except Exception as e:
        st.error(f"No pude cargar velas. Error: {e}")
        st.info("Revisa internet o el símbolo (ej: BTCUSDT).")

st.write("Tips: Para flechas, tu log debe contener BUY/SELL o COMPRA/VENTA (o ajustamos el detector).")
