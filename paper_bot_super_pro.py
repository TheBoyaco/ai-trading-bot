# -*- coding: utf-8 -*-

import os, csv, json, math, requests
import pandas as pd
print("BOOT OK paper_bot_super_pro", flush=True)
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

# =============================
# CONFIG (EDITA SOLO ESTO)
# =============================
TELEGRAM_TOKEN = "8417176005:AAE3SaZQg7BKZKiwSYyC1sjfC1Ahfe1LMSc"
TELEGRAM_CHAT_ID = "6014078078"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 800

# ðŸ”’ MODO SELECTIVO (mÃ¡s seguro, menos trades)
PROB_MIN = 0.60

START_CAPITAL = 50.0
RISK_USDT_PER_TRADE = 5.0

# ===== Filtros base =====
SESSION_UTC_START = 12
SESSION_UTC_END   = 20
USE_SESSION_FILTER = True

USE_TREND_1H_FILTER = True
USE_VOL_FILTER = True
MIN_VOL20 = 0.0010

# ===== Orderflow PRO =====
USE_ORDERFLOW_FILTER = True
BUY_RATIO_LOOKBACK = 5
BUY_RATIO_MIN = 0.53         # mÃ¡s selectivo
DELTA_BUY_MIN = 0.00         # buy_ratio_ma debe estar SUBIENDO (>= 0)
RVOL_LOOKBACK = 20
RVOL_MIN = 1.15              # volumen actual >= 1.15x promedio

# ===== Market Regime (ADX) =====
USE_ADX_FILTER = True
ADX_PERIOD = 14
ADX_MIN = 18                 # si ADX < 18 -> mercado rango/ruido: NO trade

# ===== Anti-spike =====
USE_SPIKE_FILTER = True
ATR_PERIOD = 14
SPIKE_ATR_MULT_MAX = 2.2

# ===== ConfirmaciÃ³n 2 velas =====
USE_CONFIRMATION_2 = True
CONFIRM_MAX_WAIT_BARS = 2

# ===== Salidas PRO =====
USE_ATR_TPSL = True
TP_ATR_MULT = 1.2
SL_ATR_MULT = 0.9

USE_BREAKEVEN = True
BE_ATR_MULT = 0.6

USE_PROB_DROP_EXIT = True
PROB_HARD_EXIT = 0.38
PROB_DROP_POINTS = 0.22
PROB_EXIT_CONFIRM = 2

USE_TIMEOUT_EXIT = True
MAX_HOLD_MINUTES = 60

# ===== Kill switch diario =====
USE_DAILY_KILL = True
DAILY_MAX_DRAWDOWN_PCT = 1.0   # si hoy cae -1.0% del capital -> pausa hasta maÃ±ana UTC

# ===== Racha pÃ©rdidas =====
USE_COOLDOWN = True
MAX_CONSEC_LOSSES = 2
COOLDOWN_MINUTES = 90

SEND_STATUS_EVERY_RUN = True

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_PATH = os.path.join(DATA_DIR, "paper_state.json")
LOG_PATH   = os.path.join(DATA_DIR, "paper_trades.csv")

import csv

TRADE_HEADER = [
    "ts_utc","symbol","interval","action","price","prob_up",
    "capital_before","capital_after","pnl_usd","pnl_pct","entry_price","candle_time"
]

def _ensure_trade_csv(*args, **kwargs):
    """Compat: mantiene el nombre original pero escribe CSV estable (12 columnas) usando log_trade."""
    # Intentar mapear kwargs comunes
    if kwargs:
        action = kwargs.get("action") or kwargs.get("side") or kwargs.get("tipo")
        symbol = kwargs.get("symbol") or kwargs.get("pair")
        interval = kwargs.get("interval") or kwargs.get("timeframe")
        price = kwargs.get("price")
        prob_up = kwargs.get("prob_up") or kwargs.get("prob") or kwargs.get("entry_prob")
        capital_before = kwargs.get("capital_before") or kwargs.get("cap_before")
        capital_after  = kwargs.get("capital_after") or kwargs.get("cap_after")
        pnl_usd = kwargs.get("pnl_usd") or kwargs.get("pnl") or kwargs.get("pnl_usdt")
        pnl_pct = kwargs.get("pnl_pct")
        entry_price = kwargs.get("entry_price")
        candle_time = kwargs.get("candle_time") or kwargs.get("candle") or kwargs.get("vela")
        log_trade(action=action, symbol=symbol, interval=interval, price=price, prob_up=prob_up,
                  capital_before=capital_before, capital_after=capital_after, candle_time=candle_time,
                  pnl_usd=pnl_usd, pnl_pct=pnl_pct, entry_price=entry_price)
        return

    # Si vienen posicionales, asumimos orden mÃ¡s comÃºn:
    # (action, symbol, interval, price, prob_up, cap_before, cap_after, candle_time, pnl_usd?, pnl_pct?, entry_price?)
    a = list(args)
    action = a[0] if len(a) > 0 else ""
    symbol = a[1] if len(a) > 1 else ""
    interval = a[2] if len(a) > 2 else ""
    price = a[3] if len(a) > 3 else ""
    prob_up = a[4] if len(a) > 4 else ""
    cap_before = a[5] if len(a) > 5 else ""
    cap_after  = a[6] if len(a) > 6 else ""
    candle_time = a[7] if len(a) > 7 else ""
    pnl_usd = a[8] if len(a) > 8 else ""
    pnl_pct = a[9] if len(a) > 9 else ""
    entry_price = a[10] if len(a) > 10 else ""

    log_trade(action=action, symbol=symbol, interval=interval, price=price, prob_up=prob_up,
              capital_before=cap_before, capital_after=cap_after, candle_time=candle_time,
              pnl_usd=pnl_usd, pnl_pct=pnl_pct, entry_price=entry_price)

def _write_trade_row(LOG_PATH, DATA_DIR, row):
    _ensure_trade_csv(LOG_PATH, DATA_DIR)
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)

def _coalesce(d, *keys, default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def log_trade(action=None, symbol=None, interval=None, price=None, prob_up=None,
              capital_before=None, capital_after=None, candle_time=None,
              pnl_usd="", pnl_pct="", entry_price="", **kwargs):
    """
    Logger estable: SIEMPRE 12 columnas.
    Soporta llamadas con kwargs (action=..., symbol=...) o dict via kwargs.
    """
    # Si vino todo en un dict (ej: log_trade(row=dict))
    if "row" in kwargs and isinstance(kwargs["row"], dict):
        rowd = kwargs["row"]
        action = _coalesce(rowd, "action", default=action)
        symbol = _coalesce(rowd, "symbol", default=symbol)
        interval = _coalesce(rowd, "interval", default=interval)
        price = _coalesce(rowd, "price", default=price)
        prob_up = _coalesce(rowd, "prob_up", "prob", "entry_prob", default=prob_up)
        capital_before = _coalesce(rowd, "capital_before", default=capital_before)
        capital_after  = _coalesce(rowd, "capital_after", default=capital_after)
        pnl_usd = _coalesce(rowd, "pnl_usd", "pnl", "pnl_usdt", default=pnl_usd)
        pnl_pct = _coalesce(rowd, "pnl_pct", default=pnl_pct)
        entry_price = _coalesce(rowd, "entry_price", default=entry_price)
        candle_time = _coalesce(rowd, "candle_time", default=candle_time)

    # Resolver timestamp
    try:
        ts = utc_now_iso()
    except Exception:
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    row = [
        ts,
        "" if symbol is None else str(symbol),
        "" if interval is None else str(interval),
        "" if action is None else str(action),
        "" if price is None else str(price),
        "" if prob_up is None else str(prob_up),
        "" if capital_before is None else str(capital_before),
        "" if capital_after is None else str(capital_after),
        "" if pnl_usd is None else str(pnl_usd),
        "" if pnl_pct is None else str(pnl_pct),
        "" if entry_price is None else str(entry_price),
        "" if candle_time is None else str(candle_time),
    ]
    _write_trade_row(LOG_PATH, DATA_DIR, row)


# =============================
# UTILS
# =============================
def utc_now():
    return datetime.now(timezone.utc)

def utc_str():
    return utc_now().strftime("%Y-%m-%d %H:%M UTC")

def utc_date_str():
    return utc_now().strftime("%Y-%m-%d")

def utc_hour():
    return utc_now().hour

def in_session():
    h = utc_hour()
    return SESSION_UTC_START <= h < SESSION_UTC_END

def send_telegram(msg: str):
    if (not TELEGRAM_TOKEN) or (not TELEGRAM_CHAT_ID) or ("TU_TELEGRAM" in TELEGRAM_TOKEN):
        print("[WARN] Telegram no configurado.\n" + msg)
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
            "pending_signal": None,
            # daily kill switch
            "day": utc_date_str(),
            "day_start_capital": START_CAPITAL,
            "kill_until_utc": "",
            "last_action": "INIT"
        }
    with open(STATE_PATH, "r", encoding="utf-8-sig") as f:
        st = json.load(f)
    # reset diario si cambiÃ³ el dÃ­a
    if st.get("day") != utc_date_str():
        st["day"] = utc_date_str()
        st["day_start_capital"] = float(st.get("capital", START_CAPITAL))
        st["kill_until_utc"] = ""
        st["consec_losses"] = 0
        st["prob_drop_count"] = 0
        st["pending_signal"] = None
        save_state(st)
    return st

def save_state(st: dict):
    with open(STATE_PATH, "w", encoding="utf-8-sig") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def log_trade(row: dict):
    exists = os.path.exists(LOG_PATH)
    cols = [
        "time_utc","symbol","side","entry_price","exit_price","qty",
        "pnl_usd","pnl_pct","reason","entry_prob","exit_prob",
        "tp","sl","atr","buy_ratio_ma","delta_buy","rvol","adx",
        "trend1h_ok","vol20"
    ]
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
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

def minutes_since(time_str: str) -> float:
    if not time_str:
        return 0.0
    try:
        dt = datetime.strptime(time_str.replace(" UTC",""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        return (utc_now() - dt).total_seconds() / 60.0
    except:
        return 0.0

def cooldown_active(st: dict) -> bool:
    until = st.get("cooldown_until_utc", "")
    if not until:
        return False
    try:
        dt = datetime.strptime(until.replace(" UTC",""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        return utc_now() < dt
    except:
        return False

def kill_active(st: dict) -> bool:
    until = st.get("kill_until_utc", "")
    if not until:
        return False
    try:
        dt = datetime.strptime(until.replace(" UTC",""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        return utc_now() < dt
    except:
        return False

# =============================
# DATA: Binance public API (sin API key)
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
# INDICATORS / FEATURES (aprende velas)
# =============================
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df: pd.DataFrame, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_n = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_n.replace(0, 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_n.replace(0, 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    return dx.rolling(period).mean()

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # bÃ¡sicos
    df["ret"] = df["close"].pct_change()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi14"] = rsi(df["close"], 14)
    df["vol20"] = df["ret"].rolling(20).std()
    df["atr14"] = atr(df, ATR_PERIOD)
    df["adx14"] = adx(df, ADX_PERIOD)

    # orderflow
    df["buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, 1e-9)
    df["buy_ratio_ma"] = df["buy_ratio"].rolling(BUY_RATIO_LOOKBACK).mean()
    df["delta_buy"] = df["buy_ratio_ma"].diff()

    # RVOL (volumen relativo)
    df["vol_mean"] = df["volume"].rolling(RVOL_LOOKBACK).mean()
    df["rvol"] = df["volume"] / df["vol_mean"].replace(0, 1e-9)

    # ====== FEATURES DE VELA (aprende cÃ³mo se mueven) ======
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = df["body"] / df["range"]                          # cuerpo grande = fuerza
    df["upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)).clip(lower=0)
    df["lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"]).clip(lower=0)
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"]
    df["close_pos"] = (df["close"] - df["low"]) / df["range"]            # cierre cerca del high = fuerza
    df["impulse"] = df["ret"].rolling(3).sum()                           # mini-momentum 3 velas
    df["range_atr"] = df["range"] / df["atr14"].replace(0, 1e-9)         # spike detector mÃ¡s estable

    df.dropna(inplace=True)

    # target simple: prÃ³xima vela cierre > cierre actual
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

def predict_prob(df: pd.DataFrame) -> float:
    feats = [
        "ret","ema20","ema50","rsi14","vol20","atr14",
        "buy_ratio_ma","delta_buy","rvol","adx14",
        # vela
        "body_ratio","upper_wick_ratio","lower_wick_ratio","close_pos",
        "impulse","range_atr"
    ]
    X = df[feats]
    y = df["target"]

    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]

    model = RandomForestClassifier(
        n_estimators=450,
        random_state=42,
        max_depth=10,
        min_samples_leaf=5
    )
    import numpy as _np

    y_train = _np.array(y_train)

    y_train = _np.where(_np.isnan(y_train), 0, y_train).astype(int)

    if _np.unique(y_train).size < 2: return 0.5

    model.fit(X_train, y_train)
    return float(model.predict_proba(X.iloc[-1:])[0][1])

# =============================
# FILTERS
# =============================
def trend1h_ok(symbol: str) -> bool:
    df1h = get_klines(symbol, "1h", 350)
    df1h = prepare(df1h)
    last = df1h.iloc[-1]
    return (last["close"] > last["ema50"]) and (last["ema20"] > last["ema50"])

def spike_ok(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    # usamos range_atr para ser robustos
    return float(last["range_atr"]) <= SPIKE_ATR_MULT_MAX

# =============================
# PAPER EXECUTION
# =============================
def calc_qty(capital: float, price: float) -> float:
    use = min(RISK_USDT_PER_TRADE, capital)
    return max(use / price, 0.0)

def enter_trade(st: dict, price: float, prob: float, lastrow: pd.Series, trend_ok_1h: bool):
    qty = calc_qty(float(st["capital"]), price)
    atr_val = float(lastrow["atr14"])
    if qty <= 0 or atr_val <= 0:
        return False

    tp = price + (TP_ATR_MULT * atr_val)
    sl = price - (SL_ATR_MULT * atr_val)

    st["in_position"] = True
    st["entry_price"] = price
    st["qty"] = qty
    st["tp"] = tp
    st["sl"] = sl
    st["entry_time_utc"] = utc_str()
    st["entry_prob"] = prob
    st["prob_drop_count"] = 0
    st["pending_signal"] = None
    st["last_action"] = "ENTRY"
    save_state(st)

    send_telegram(
        "ENTRADA (SIMULADA)\n"
        f"{SYMBOL} {INTERVAL}\n"
        f"Precio: {price:,.2f}\n"
        f"Prob: {prob*100:.1f}% (min {PROB_MIN*100:.0f}%)\n"
        f"TP: {tp:,.2f} | SL: {sl:,.2f}\n"
        f"ATR: {atr_val:,.2f} | ADX: {float(lastrow['adx14']):.1f}\n"
        f"buy_ratio_ma: {float(lastrow['buy_ratio_ma']):.2f} | Î”buy: {float(lastrow['delta_buy']):+.3f}\n"
        f"RVOL: {float(lastrow['rvol']):.2f} | Vela body%: {float(lastrow['body_ratio']):.2f} | close_pos: {float(lastrow['close_pos']):.2f}\n"
        f"Tendencia 1H: {'OK' if trend_ok_1h else 'NO'}\n"
        f"Hora: {utc_str()}"
    )
    return True

def exit_trade(st: dict, exit_price: float, exit_prob: float, reason: str, lastrow: pd.Series, trend_ok_1h: bool):
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

    # racha pÃ©rdidas
    if pnl_usd < 0:
        st["consec_losses"] = int(st.get("consec_losses", 0)) + 1
    else:
        st["consec_losses"] = 0

    if USE_COOLDOWN and st["consec_losses"] >= MAX_CONSEC_LOSSES:
        until = utc_now() + pd.Timedelta(minutes=COOLDOWN_MINUTES)
        st["cooldown_until_utc"] = until.strftime("%Y-%m-%d %H:%M UTC")

    # daily kill switch
    if USE_DAILY_KILL:
        day_start = float(st.get("day_start_capital", START_CAPITAL))
        dd_pct = (st["capital"] - day_start) / day_start * 100.0
        if dd_pct <= -DAILY_MAX_DRAWDOWN_PCT:
            # pausar hasta maÃ±ana 00:05 UTC
            tomorrow = (utc_now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            st["kill_until_utc"] = f"{tomorrow} 00:05 UTC"

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
        "entry_prob": f"{float(st.get('entry_prob', 0.0)):.4f}",
        "exit_prob": f"{exit_prob:.4f}",
        "tp": f"{0.0:.2f}",
        "sl": f"{0.0:.2f}",
        "atr": f"{float(lastrow['atr14']):.2f}",
        "buy_ratio_ma": f"{float(lastrow['buy_ratio_ma']):.4f}",
        "delta_buy": f"{float(lastrow['delta_buy']):.4f}",
        "rvol": f"{float(lastrow['rvol']):.4f}",
        "adx": f"{float(lastrow['adx14']):.2f}",
        "trend1h_ok": str(trend_ok_1h),
        "vol20": f"{float(lastrow['vol20']):.6f}",
    })

    send_telegram(
        "SALIDA (SIMULADA)\n"
        f"{SYMBOL} {INTERVAL}\n"
        f"Precio salida: {exit_price:,.2f}\n"
        f"PnL: {pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)\n"
        f"Motivo: {reason}\n"
        f"Prob salida: {exit_prob*100:.1f}%\n"
        f"Capital: ${st['capital']:.2f}\n"
        f"Cooldown: {st.get('cooldown_until_utc','') or 'NO'}\n"
        f"Kill diario: {st.get('kill_until_utc','') or 'NO'}\n"
        f"Hora: {utc_str()}"
    )

# =============================
# RUN ONCE
# =============================
def run_once():
    print('CP1 load_state()', flush=True)
    st = load_state()

    print('CP2 get_klines()', flush=True)
    df_raw = get_klines(SYMBOL, INTERVAL, LIMIT)
    print('CP3 prepare()', flush=True)
    df = prepare(df_raw)
    last = df.iloc[-1]
    price = float(last["close"])
    print('CP4 predict_prob()', flush=True)
    prob  = predict_prob(df)

    # tendencia 1H
    trend_ok_1h = True
    if USE_TREND_1H_FILTER:
        try:
            trend_ok_1h = trend1h_ok(SYMBOL)
        except Exception:
            trend_ok_1h = True

    # -------- EXIT ----------
    if st.get("in_position", False):
        entry = float(st["entry_price"])
        tp = float(st["tp"])
        sl = float(st["sl"])
        entry_prob = float(st.get("entry_prob", 0.0))

        # TP / SL
        if price >= tp and tp > 0:
            exit_trade(st, price, prob, "TP", last, trend_ok_1h); return
        if price <= sl and sl > 0:
            exit_trade(st, price, prob, "SL", last, trend_ok_1h); return

        # Break-even
        if USE_BREAKEVEN and float(last["atr14"]) > 0:
            if price >= entry + (BE_ATR_MULT * float(last["atr14"])):
                if st["sl"] < entry:
                    st["sl"] = entry
                    save_state(st)
                    send_telegram(f"GESTIÃ“N (SIMULADA)\nBreak-even âœ… SL -> {entry:,.2f}\nHora: {utc_str()}")

        # Prob drop exit
        if USE_PROB_DROP_EXIT:
            drop_points = entry_prob - prob
            if (prob <= PROB_HARD_EXIT) or (drop_points >= PROB_DROP_POINTS):
                st["prob_drop_count"] = int(st.get("prob_drop_count", 0)) + 1
            else:
                st["prob_drop_count"] = 0
            save_state(st)

            if st["prob_drop_count"] >= PROB_EXIT_CONFIRM:
                exit_trade(st, price, prob, "PROB_DROP", last, trend_ok_1h); return

        # Timeout
        if USE_TIMEOUT_EXIT and minutes_since(st.get("entry_time_utc","")) >= MAX_HOLD_MINUTES:
            exit_trade(st, price, prob, "TIMEOUT", last, trend_ok_1h); return

    # -------- ENTRY ----------
    reasons = []
    entry_allowed = True

    if st.get("in_position", False):
        entry_allowed = False; reasons.append("Ya en posiciÃ³n")

    if USE_DAILY_KILL and kill_active(st):
        entry_allowed = False; reasons.append(f"KILL hasta {st.get('kill_until_utc')}")

    if USE_COOLDOWN and cooldown_active(st):
        entry_allowed = False; reasons.append(f"COOLDOWN hasta {st.get('cooldown_until_utc')}")

    if USE_SESSION_FILTER and not in_session():
        entry_allowed = False; reasons.append(f"Fuera horario UTC {SESSION_UTC_START}-{SESSION_UTC_END}")

    if USE_TREND_1H_FILTER and not trend_ok_1h:
        entry_allowed = False; reasons.append("Tendencia 1H NO")

    if USE_VOL_FILTER and float(last["vol20"]) < MIN_VOL20:
        entry_allowed = False; reasons.append("Volatilidad baja")

    if USE_ADX_FILTER and float(last["adx14"]) < ADX_MIN:
        entry_allowed = False; reasons.append(f"ADX bajo ({float(last['adx14']):.1f}<{ADX_MIN})")

    if USE_ORDERFLOW_FILTER:
        if float(last["buy_ratio_ma"]) < BUY_RATIO_MIN:
            entry_allowed = False; reasons.append(f"buy_ratio bajo ({float(last['buy_ratio_ma']):.2f}<{BUY_RATIO_MIN:.2f})")
        if float(last["delta_buy"]) < DELTA_BUY_MIN:
            entry_allowed = False; reasons.append(f"Î”buy negativo ({float(last['delta_buy']):+.3f})")
        if float(last["rvol"]) < RVOL_MIN:
            entry_allowed = False; reasons.append(f"RVOL bajo ({float(last['rvol']):.2f}<{RVOL_MIN:.2f})")

    if USE_SPIKE_FILTER and not spike_ok(df):
        entry_allowed = False; reasons.append("Spike vela (rango/ATR alto)")

    if prob < PROB_MIN:
        entry_allowed = False; reasons.append(f"Prob {prob*100:.1f}% < {PROB_MIN*100:.0f}%")

    # ConfirmaciÃ³n 2 velas
    if USE_CONFIRMATION_2:
        pending = st.get("pending_signal", None)

        if entry_allowed:
            if pending is None:
                st["pending_signal"] = {"created_time": utc_str(), "bars_waited": 0}
                st["last_action"] = "PENDING"
                save_state(st)
                if SEND_STATUS_EVERY_RUN:
                    send_telegram(
                        "STATUS\n"
                        f"{SYMBOL} {INTERVAL}\n"
                        f"Precio: {price:,.2f}\n"
                        f"Prob: {prob*100:.1f}%\n"
                        f"Pre-seÃ±al âœ… esperando confirmaciÃ³n\n"
                        f"ADX:{float(last['adx14']):.1f} RVOL:{float(last['rvol']):.2f} buy:{float(last['buy_ratio_ma']):.2f} Î”buy:{float(last['delta_buy']):+.3f}\n"
                        f"Hora: {utc_str()}"
                    )
                return
            else:
                pending["bars_waited"] = int(pending.get("bars_waited", 0)) + 1
                st["pending_signal"] = pending
                save_state(st)
                enter_trade(st, price, prob, last, trend_ok_1h)
                st["pending_signal"] = None
                save_state(st)
                return
        else:
            if pending is not None:
                pending["bars_waited"] = int(pending.get("bars_waited", 0)) + 1
                if pending["bars_waited"] >= CONFIRM_MAX_WAIT_BARS:
                    st["pending_signal"] = None
                else:
                    st["pending_signal"] = pending
                save_state(st)

    else:
        if entry_allowed:
            enter_trade(st, price, prob, last, trend_ok_1h)
            return

    # STATUS
    if SEND_STATUS_EVERY_RUN:
        pos = "SI" if st.get("in_position", False) else "NO"
        reason_txt = " | ".join(reasons) if reasons else "OK"
        send_telegram(
            "STATUS\n"
            f"{SYMBOL} {INTERVAL}\n"
            f"Precio: {price:,.2f}\n"
            f"Prob: {prob*100:.1f}% (min {PROB_MIN*100:.0f}%)\n"
            f"En posiciÃ³n: {pos} | Capital: ${float(st.get('capital', START_CAPITAL)):.2f}\n"
            f"ADX:{float(last['adx14']):.1f} RVOL:{float(last['rvol']):.2f} buy:{float(last['buy_ratio_ma']):.2f} Î”buy:{float(last['delta_buy']):+.3f}\n"
            f"Vela body%:{float(last['body_ratio']):.2f} close_pos:{float(last['close_pos']):.2f} range/ATR:{float(last['range_atr']):.2f}\n"
            f"Tendencia 1H: {'OK' if trend_ok_1h else 'NO'} | Vol20:{float(last['vol20']):.4f}\n"
            f"Filtros: {reason_txt}\n"
            f"Hora: {utc_str()}"
        )

    
    run_once()
run_once()









if __name__ == '__main__':
    run_once()


