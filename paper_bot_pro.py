# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import math
import requests
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

# =============================
# CONFIG (EDITA SOLO ESTO)
# =============================
TELEGRAM_TOKEN = "8417176005:AAE3SaZQg7BKZKiwSYyC1sjfC1Ahfe1LMSc"
TELEGRAM_CHAT_ID = "6014078078"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 600  # velas para entrenar y features

# Umbral IA
PROB_MIN = 0.55

# CAPITAL SIMULADO
START_CAPITAL = 50.0
RISK_USDT_PER_TRADE = 50.0  # en paper puedes usar todo el capital

# ===== Filtros =====
# Horario UTC recomendado (puedes ajustar)
SESSION_UTC_START = 12
SESSION_UTC_END = 20

USE_SESSION_FILTER = True
USE_TREND_1H_FILTER = True
USE_VOL_FILTER = True
USE_ORDERFLOW_FILTER = True
USE_SPIKE_FILTER = True
USE_CONFIRMATION_2 = True

# Volatilidad mínima (evita mercado plano)
MIN_VOL20 = 0.0010

# Presión compradora: promedio últimas N velas
BUY_RATIO_LOOKBACK = 5
BUY_RATIO_MIN = 0.52

# Anti-spike: evita entrar si última vela fue demasiado grande
SPIKE_ATR_MULT_MAX = 2.2

# Confirmación 2 velas:
# - primera señal arma "pre-senal"
# - si en el próximo check sigue válida -> entra
CONFIRM_MAX_WAIT_BARS = 2

# ===== Salidas pro =====
USE_ATR_TPSL = True
ATR_PERIOD = 14
TP_ATR_MULT = 1.2
SL_ATR_MULT = 0.9

# Break-even: cuando el precio avanza a favor X ATR, sube SL a entrada
USE_BREAKEVEN = True
BE_ATR_MULT = 0.6

# Salida por caída de probabilidad (muy importante para evitar pérdidas como la de 89,455 -> 88,880)
USE_PROB_DROP_EXIT = True
PROB_HARD_EXIT = 0.35   # si baja de aquí
PROB_DROP_POINTS = 0.25 # o si cae 25 puntos vs prob al entrar (ej 0.77->0.52)
PROB_EXIT_CONFIRM = 2   # 2 checks seguidos

# Timeout (si no pega TP rápido, mejor salir)
USE_TIMEOUT_EXIT = True
MAX_HOLD_MINUTES = 60  # 4 velas de 15m

# Bloqueo por racha de pérdidas
USE_COOLDOWN = True
MAX_CONSEC_LOSSES = 2
COOLDOWN_MINUTES = 90

# Reportes
SEND_STATUS_EVERY_RUN = True

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_PATH = os.path.join(DATA_DIR, "paper_state.json")
LOG_PATH = os.path.join(DATA_DIR, "paper_trades.csv")

# =============================
# UTILS
# =============================
def utc_now():
    return datetime.now(timezone.utc)

def utc_str():
    return utc_now().strftime("%Y-%m-%d %H:%M UTC")

def utc_hour():
    return utc_now().hour

def in_session():
    h = utc_hour()
    return SESSION_UTC_START <= h < SESSION_UTC_END

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or "TU_TELEGRAM" in TELEGRAM_TOKEN:
        print("[WARN] Telegram no configurado.")
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=25)

def load_state():
    if not os.path.exists(STATE_PATH):
        return {
            "capital": START_CAPITAL,
            "in_position": False,
            "entry_price": 0.0,
            "qty": 0.0,
            "tp": 0.0,
            "sl": 0.0,
            "entry_time_utc": "",
            "entry_prob": 0.0,
            "prob_drop_count": 0,
            "consec_losses": 0,
            "cooldown_until_utc": "",
            "pending_signal": None,  # {"created_time":..., "bars_waited":..., "prob":..., "price":...}
            "last_action": "INIT"
        }
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(st: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def log_trade(row: dict):
    exists = os.path.exists(LOG_PATH)
    cols = [
        "time_utc","symbol","side","entry_price","exit_price","qty",
        "pnl_usd","pnl_pct","reason","entry_prob","exit_prob",
        "tp","sl","atr","buy_ratio_ma","trend1h_ok","vol20"
    ]
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})

def parse_interval_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    return 15

# =============================
# DATA: Binance public API
# =============================
def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=25)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","taker_buy_base"]:
        df[c] = df[c].astype(float)
    return df

# =============================
# INDICATORS
# =============================
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi14"] = rsi(df["close"], 14)
    df["vol20"] = df["ret"].rolling(20).std()
    df["atr14"] = atr(df, ATR_PERIOD)

    df["buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, 1e-9)
    df["buy_ratio_ma"] = df["buy_ratio"].rolling(BUY_RATIO_LOOKBACK).mean()

    df.dropna(inplace=True)

    # target: próxima vela sube (clasificación simple)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

def predict_prob(df: pd.DataFrame) -> float:
    feats = ["ret","ema20","ema50","rsi14","vol20","atr14","buy_ratio_ma"]
    X = df[feats]
    y = df["target"]

    # split temporal (sin shuffle)
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]

    model = RandomForestClassifier(
        n_estimators=350,
        random_state=42,
        max_depth=9,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    return float(model.predict_proba(X.iloc[-1:])[0][1])

# =============================
# FILTERS
# =============================
def trend1h_ok(symbol: str) -> bool:
    # permiso: 1H alcista básico (close > ema50 y ema20>ema50)
    df1h = get_klines(symbol, "1h", 300)
    df1h = prepare(df1h)
    last = df1h.iloc[-1]
    return (last["close"] > last["ema50"]) and (last["ema20"] > last["ema50"])

def vol_ok(df: pd.DataFrame) -> bool:
    return float(df["vol20"].iloc[-1]) >= MIN_VOL20

def orderflow_ok(df: pd.DataFrame) -> bool:
    return float(df["buy_ratio_ma"].iloc[-1]) >= BUY_RATIO_MIN

def spike_ok(df: pd.DataFrame) -> bool:
    # última vela no debe ser un spike enorme vs ATR
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    a = float(last["atr14"])
    if a <= 0:
        return True
    return rng <= (SPIKE_ATR_MULT_MAX * a)

# =============================
# POSITION MANAGEMENT (PAPER)
# =============================
def calc_qty(capital: float, price: float) -> float:
    use = min(RISK_USDT_PER_TRADE, capital)
    qty = use / price
    return max(qty, 0.0)

def enter_trade(st: dict, price: float, prob: float, atr_val: float, buy_ratio_ma: float, trend_ok_1h: bool, vol20: float):
    qty = calc_qty(st["capital"], price)
    if qty <= 0:
        return False

    tp = sl = 0.0
    if USE_ATR_TPSL:
        tp = price + (TP_ATR_MULT * atr_val)
        sl = price - (SL_ATR_MULT * atr_val)
    else:
        # fallback (no recomendado)
        tp = price * (1 + 0.003)
        sl = price * (1 - 0.002)

    st["in_position"] = True
    st["entry_price"] = price
    st["qty"] = qty
    st["tp"] = tp
    st["sl"] = sl
    st["entry_time_utc"] = utc_str()
    st["entry_prob"] = prob
    st["prob_drop_count"] = 0
    st["last_action"] = "ENTRY"
    save_state(st)

    send_telegram(
        "ENTRADA (SIMULADA)\n"
        f"{SYMBOL} {INTERVAL}\n"
        f"Precio: {price:,.2f}\n"
        f"Prob: {prob*100:.1f}%\n"
        f"TP: {tp:,.2f}\n"
        f"SL: {sl:,.2f}\n"
        f"ATR: {atr_val:,.2f}\n"
        f"buy_ratio_ma: {buy_ratio_ma:.2f}\n"
        f"Tendencia 1H: {'OK' if trend_ok_1h else 'NO'}\n"
        f"Vol20: {vol20:.4f}\n"
        f"Hora: {utc_str()}"
    )
    return True

def exit_trade(st: dict, exit_price: float, exit_prob: float, reason: str, atr_val: float, buy_ratio_ma: float, trend_ok_1h: bool, vol20: float):
    entry = float(st["entry_price"])
    qty = float(st["qty"])
    if entry <= 0 or qty <= 0:
        return

    pnl_usd = (exit_price - entry) * qty
    pnl_pct = (exit_price - entry) / entry * 100.0

    st["capital"] = float(st["capital"]) + pnl_usd
    st["in_position"] = False
    st["entry_price"] = 0.0
    st["qty"] = 0.0
    st["tp"] = 0.0
    st["sl"] = 0.0
    st["entry_time_utc"] = ""
    st["last_action"] = "EXIT"

    # racha pérdidas + cooldown
    if pnl_usd < 0:
        st["consec_losses"] = int(st.get("consec_losses", 0)) + 1
    else:
        st["consec_losses"] = 0

    if USE_COOLDOWN and st["consec_losses"] >= MAX_CONSEC_LOSSES:
        until = utc_now() + pd.Timedelta(minutes=COOLDOWN_MINUTES)
        st["cooldown_until_utc"] = until.strftime("%Y-%m-%d %H:%M UTC")

    save_state(st)

    log_trade({
        "time_utc": utc_str(),
        "symbol": SYMBOL,
        "side": "LONG",
        "entry_price": f"{entry:.2f}",
        "exit_price": f"{exit_price:.2f}",
        "qty": f"{qty:.6f}",
        "pnl_usd": f"{pnl_usd:.4f}",
        "pnl_pct": f"{pnl_pct:.4f}",
        "reason": reason,
        "entry_prob": f"{st.get('entry_prob', 0.0):.4f}",
        "exit_prob": f"{exit_prob:.4f}",
        "tp": f"{st.get('tp', 0.0):.2f}",
        "sl": f"{st.get('sl', 0.0):.2f}",
        "atr": f"{atr_val:.2f}",
        "buy_ratio_ma": f"{buy_ratio_ma:.4f}",
        "trend1h_ok": str(trend_ok_1h),
        "vol20": f"{vol20:.6f}",
    })

    send_telegram(
        "SALIDA (SIMULADA)\n"
        f"{SYMBOL} {INTERVAL}\n"
        f"Precio salida: {exit_price:,.2f}\n"
        f"PnL: {pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)\n"
        f"Motivo: {reason}\n"
        f"Prob salida: {exit_prob*100:.1f}%\n"
        f"Capital actual: ${st['capital']:.2f}\n"
        f"Hora: {utc_str()}"
    )

# =============================
# MAIN LOOP (1 RUN)
# =============================
def cooldown_active(st: dict) -> bool:
    until = st.get("cooldown_until_utc", "")
    if not until:
        return False
    try:
        # parse simple "YYYY-mm-dd HH:MM UTC"
        dt = datetime.strptime(until.replace(" UTC",""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        return utc_now() < dt
    except:
        return False

def minutes_since(time_str: str) -> float:
    if not time_str:
        return 0.0
    try:
        dt = datetime.strptime(time_str.replace(" UTC",""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        delta = utc_now() - dt
        return delta.total_seconds() / 60.0
    except:
        return 0.0

def run_once():
    st = load_state()

    # Data 15m
    df_raw = get_klines(SYMBOL, INTERVAL, LIMIT)
    df = prepare(df_raw)

    price = float(df["close"].iloc[-1])
    prob = predict_prob(df)

    atr_val = float(df["atr14"].iloc[-1])
    buy_ratio_ma = float(df["buy_ratio_ma"].iloc[-1])
    vol20 = float(df["vol20"].iloc[-1])

    # Trend 1H
    trend_ok_1h = True
    if USE_TREND_1H_FILTER:
        try:
            trend_ok_1h = trend1h_ok(SYMBOL)
        except Exception as e:
            trend_ok_1h = True  # si falla, no bloquea (para no romper bot)
            print("[WARN] trend1h error:", e)

    # ======================
    # EXIT LOGIC
    # ======================
    if st.get("in_position", False):
        entry = float(st["entry_price"])
        tp = float(st["tp"])
        sl = float(st["sl"])
        entry_prob = float(st.get("entry_prob", 0.0))

        # 1) TP / SL
        if tp > 0 and price >= tp:
            exit_trade(st, price, prob, "TP", atr_val, buy_ratio_ma, trend_ok_1h, vol20)
            return
        if sl > 0 and price <= sl:
            exit_trade(st, price, prob, "SL", atr_val, buy_ratio_ma, trend_ok_1h, vol20)
            return

        # 2) Break-even
        if USE_BREAKEVEN and atr_val > 0 and entry > 0:
            # cuando avance BE_ATR_MULT*ATR a favor, sube SL a entrada
            if price >= entry + (BE_ATR_MULT * atr_val):
                if st["sl"] < entry:
                    st["sl"] = entry
                    save_state(st)
                    send_telegram(
                        "GESTIÓN (SIMULADA)\n"
                        f"Break-even activado ✅\n"
                        f"SL movido a entrada: {entry:,.2f}\n"
                        f"Hora: {utc_str()}"
                    )

        # 3) Salida por caída de probabilidad
        if USE_PROB_DROP_EXIT:
            drop_points = entry_prob - prob
            if (prob <= PROB_HARD_EXIT) or (drop_points >= PROB_DROP_POINTS):
                st["prob_drop_count"] = int(st.get("prob_drop_count", 0)) + 1
            else:
                st["prob_drop_count"] = 0
            save_state(st)

            if st["prob_drop_count"] >= PROB_EXIT_CONFIRM:
                exit_trade(st, price, prob, "PROB_DROP", atr_val, buy_ratio_ma, trend_ok_1h, vol20)
                return

        # 4) Timeout
        if USE_TIMEOUT_EXIT:
            held_min = minutes_since(st.get("entry_time_utc", ""))
            if held_min >= MAX_HOLD_MINUTES:
                exit_trade(st, price, prob, "TIMEOUT", atr_val, buy_ratio_ma, trend_ok_1h, vol20)
                return

    # ======================
    # ENTRY LOGIC
    # ======================
    reasons = []
    entry_allowed = True

    if st.get("in_position", False):
        entry_allowed = False
        reasons.append("Ya en posición")

    if USE_COOLDOWN and cooldown_active(st):
        entry_allowed = False
        reasons.append(f"COOLDOWN hasta {st.get('cooldown_until_utc')}")

    if USE_SESSION_FILTER and not in_session():
        entry_allowed = False
        reasons.append(f"Fuera horario UTC {SESSION_UTC_START}-{SESSION_UTC_END}")

    if USE_TREND_1H_FILTER and not trend_ok_1h:
        entry_allowed = False
        reasons.append("Tendencia 1H NO")

    if USE_VOL_FILTER and not vol_ok(df):
        entry_allowed = False
        reasons.append("Volatilidad baja")

    if USE_ORDERFLOW_FILTER and not orderflow_ok(df):
        entry_allowed = False
        reasons.append(f"Presión compra baja ({buy_ratio_ma:.2f}<{BUY_RATIO_MIN:.2f})")

    if USE_SPIKE_FILTER and not spike_ok(df):
        entry_allowed = False
        reasons.append("Spike vela (entrada tarde)")

    # Prob mínima
    if prob < PROB_MIN:
        entry_allowed = False
        reasons.append(f"Prob {prob*100:.1f}% < {PROB_MIN*100:.0f}%")

    # Confirmación 2 velas
    if USE_CONFIRMATION_2:
        pending = st.get("pending_signal", None)

        if entry_allowed:
            if pending is None:
                # crea pre-señal
                st["pending_signal"] = {
                    "created_time": utc_str(),
                    "bars_waited": 0,
                    "prob": prob,
                    "price": price
                }
                st["last_action"] = "PENDING"
                save_state(st)

                if SEND_STATUS_EVERY_RUN:
                    send_telegram(
                        "STATUS\n"
                        f"{SYMBOL} {INTERVAL}\n"
                        f"Precio: {price:,.2f}\n"
                        f"Prob: {prob*100:.1f}%\n"
                        f"Pre-señal creada ✅ (esperando confirmación)\n"
                        f"Hora: {utc_str()}"
                    )
                return
            else:
                # ya había pre-señal: confirma
                pending["bars_waited"] = int(pending.get("bars_waited", 0)) + 1
                st["pending_signal"] = pending
                save_state(st)

                # confirma y entra
                enter_trade(st, price, prob, atr_val, buy_ratio_ma, trend_ok_1h, vol20)
                st["pending_signal"] = None
                save_state(st)
                return

        else:
            # si no hay condiciones, aumenta espera o borra pre-señal
            if pending is not None:
                pending["bars_waited"] = int(pending.get("bars_waited", 0)) + 1
                if pending["bars_waited"] >= CONFIRM_MAX_WAIT_BARS:
                    st["pending_signal"] = None
                else:
                    st["pending_signal"] = pending
                save_state(st)

    else:
        # sin confirmación: entra directo
        if entry_allowed:
            enter_trade(st, price, prob, atr_val, buy_ratio_ma, trend_ok_1h, vol20)
            return

    # ======================
    # STATUS
    # ======================
    if SEND_STATUS_EVERY_RUN:
        pos = "SI" if st.get("in_position", False) else "NO"
        reason_txt = " | ".join(reasons) if reasons else "OK"
        send_telegram(
            "STATUS\n"
            f"{SYMBOL} {INTERVAL}\n"
            f"Precio: {price:,.2f}\n"
            f"Prob: {prob*100:.1f}%\n"
            f"En posición: {pos}\n"
            f"Capital: ${st.get('capital', START_CAPITAL):.2f}\n"
            f"buy_ratio_ma: {buy_ratio_ma:.2f}\n"
            f"Tendencia 1H: {'OK' if trend_ok_1h else 'NO'}\n"
            f"Vol20: {vol20:.4f}\n"
            f"Filtros: {reason_txt}\n"
            f"Hora: {utc_str()}"
        )

    print(f"{utc_str()} | price={price:.2f} prob={prob:.3f} pos={pos} cap={st.get('capital'):.2f}")

if __name__ == "__main__":
    run_once()
