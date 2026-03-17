# -*- coding: utf-8 -*-

import os
import json
import csv
import requests
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ====== TELEGRAM ======
TELEGRAM_TOKEN = "8417176005:AAE3SaZQg7BKZKiwSYyC1sjfC1Ahfe1LMSc"
TELEGRAM_CHAT_ID = "6014078078"

# ====== CONFIG ======
SYMBOL = "BTCUSDT"
INTERVAL = "15m"            # mas oportunidades
LIMIT = 300

CAPITAL_INICIAL = 50.0      # USD simulados
RIESGO_POR_TRADE = 0.005    # 0.5% del capital por trade (conservador)
TAKE_PROFIT = 0.003         # +0.3%
STOP_LOSS = 0.002           # -0.2%
PROB_MIN = 0.55             # baja a 0.50 si quieres mas entradas

ENVIAR_STATUS_CADA_EJECUCION = True  # mas mensajes a Telegram

# ====== RUTAS ======
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_PATH = os.path.join(DATA_DIR, "paper_state.json")
TRADES_CSV = os.path.join(DATA_DIR, "paper_trades.csv")

# ====== TELEGRAM (DESDE config.json) ======
import json

CONFIG_PATH = r"C:\bot_ia_trading\config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)

_cfg = load_config()
TELEGRAM_TOKEN = _cfg.get("telegram_token", "").strip()
TELEGRAM_CHAT_ID = str(_cfg.get("telegram_chat_id", "")).strip()
SEND_STATUS_EVERY_RUN = bool(_cfg.get("send_status_every_run", True))


def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=15
        )
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text)
    except Exception as e:
        print("Telegram exception:", e)

# ====== ESTADO PERSISTENTE ======
def load_state():
    if not os.path.exists(STATE_PATH):
        return {
            "capital": CAPITAL_INICIAL,
            "en_posicion": False,
            "precio_entrada": 0.0,
            "cantidad": 0.0,
            "ultima_accion": "NONE",
            "updated_utc": ""
        }
    with open(STATE_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def save_state(state: dict):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ====== HISTORIAL TRADES ======
def log_trade(row: dict):
    existe = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existe:
            w.writeheader()
        w.writerow(row)

# ====== BINANCE DATA ======
def get_klines():
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}, timeout=15)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "_","_","_","_","_","_"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["vol20"] = df["volume"].rolling(20).mean()
    df.dropna(inplace=True)
    df["target"] = (df["ret"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df

def prob_subida(df: pd.DataFrame) -> float:
    X = df[["ret","sma20","sma50","vol20"]]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return float(model.predict_proba(X.iloc[-1:])[0][1])

# ====== BOT PAPER ======
def run():
    state = load_state()

    df = preparar(get_klines())
    precio_actual = float(df["close"].iloc[-1])
    candle_time = str(df["open_time"].iloc[-1])
    prob = prob_subida(df)

    hora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    capital = float(state["capital"])
    en_posicion = bool(state["en_posicion"])
    precio_entrada = float(state["precio_entrada"])
    cantidad = float(state["cantidad"])

    accion = "STATUS"

    # ====== ENTRADA ======
    if (not en_posicion) and (prob >= PROB_MIN):
        riesgo_usd = capital * RIESGO_POR_TRADE
        cantidad = riesgo_usd / precio_actual
        precio_entrada = precio_actual
        en_posicion = True
        accion = "ENTRADA"

        log_trade({
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "action": "ENTRY",
            "price": precio_entrada,
            "prob_up": prob,
            "capital_before": capital,
            "capital_after": capital,
            "candle_time": candle_time
        })

        send_telegram(
            "ENTRADA (SIMULADA)\n"
            f"Precio: {precio_entrada:,.2f}\n"
            f"Prob subida: {prob*100:.1f}%\n"
            f"Capital: ${capital:.2f}\n"
            f"Hora: {hora}"
        )

    # ====== SALIDA ======
    elif en_posicion:
        cambio = (precio_actual - precio_entrada) / precio_entrada

        if (cambio >= TAKE_PROFIT) or (cambio <= -STOP_LOSS):
            pnl = cambio * capital
            capital_nuevo = capital + pnl
            accion = "SALIDA"

            log_trade({
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "action": "EXIT",
                "price": precio_actual,
                "prob_up": prob,
                "capital_before": capital,
                "capital_after": capital_nuevo,
                "pnl_usd": pnl,
                "pnl_pct": cambio * 100.0,
                "entry_price": precio_entrada,
                "candle_time": candle_time
            })

            capital = capital_nuevo
            en_posicion = False
            precio_entrada = 0.0
            cantidad = 0.0

            send_telegram(
                "SALIDA (SIMULADA)\n"
                f"Precio salida: {precio_actual:,.2f}\n"
                f"PnL: {pnl:+.2f} USD\n"
                f"Capital actual: ${capital:.2f}\n"
                f"Hora: {hora}"
            )

    # ====== STATUS SIEMPRE (para que te mande mas mensajes) ======
    if ENVIAR_STATUS_CADA_EJECUCION:
        pos_txt = "SI" if en_posicion else "NO"
        send_telegram(
            "STATUS\n"
            f"{SYMBOL} {INTERVAL}\n"
            f"Precio: {precio_actual:,.2f}\n"
            f"Prob subida: {prob*100:.1f}%\n"
            f"En posicion: {pos_txt}\n"
            f"Capital: ${capital:.2f}\n"
            f"Hora: {hora}"
        )

    # ====== PRINT PARA LOGS ======
    print(f"{hora} | {SYMBOL} | precio {precio_actual:,.2f} | prob {prob*100:.1f}% | capital {capital:.2f} | en_posicion {en_posicion} | accion {accion}")

    # ====== GUARDAR ESTADO ======
    state["capital"] = capital
    state["en_posicion"] = en_posicion
    state["precio_entrada"] = precio_entrada
    state["cantidad"] = cantidad
    state["ultima_accion"] = accion
    save_state(state)

if __name__ == "__main__":
    run()

