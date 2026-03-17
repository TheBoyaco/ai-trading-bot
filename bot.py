# -*- coding: utf-8 -*-

import requests
import pandas as pd
import os
import csv
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ================= CONFIGURACIÓN =================

TELEGRAM_TOKEN = "8417176005:AAE3SaZQg7BKZKiwSYyC1sjfC1Ahfe1LMSc"
TELEGRAM_CHAT_ID = "6014078078"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
LIMIT = 300
PROB_THRESHOLD = 0.55

# ================= RUTAS =================

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
SIGNALS_CSV = os.path.join(DATA_DIR, "signals.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ================= TELEGRAM =================

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
        timeout=15
    )

# ================= HISTORIAL =================

def guardar_historial(symbol, price, prob, candle_time):
    existe = os.path.exists(SIGNALS_CSV)
    with open(SIGNALS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow([
                "ts", "symbol", "interval",
                "candle_time", "price", "prob_up"
            ])
        writer.writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            symbol,
            INTERVAL,
            candle_time,
            price,
            prob
        ])

# ================= BINANCE =================

def get_klines(symbol):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": LIMIT
        },
        timeout=15
    )
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "_", "_", "_", "_", "_", "_"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df

# ================= FEATURES =================

def preparar_datos(df):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["vol20"] = df["volume"].rolling(20).mean()
    df.dropna(inplace=True)
    df["target"] = (df["ret"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df

# ================= IA =================

def entrenar_y_predecir(df):
    X = df[["ret", "sma20", "sma50", "vol20"]]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    prob = model.predict_proba(X.iloc[-1:])[0][1]
    return prob

# ================= MAIN =================

def run():
    for symbol in SYMBOLS:
        df = get_klines(symbol)
        df = preparar_datos(df)

        prob = entrenar_y_predecir(df)
        price = df["close"].iloc[-1]
        candle_time = df["open_time"].iloc[-1]

        guardar_historial(symbol, price, prob, candle_time)

        print(f"{symbol} | Precio: {price:,.2f} | Prob subida: {prob*100:.1f}%")

        if prob >= PROB_THRESHOLD:
            msg = (
                f"{symbol}\n"
                f"Precio: {price:,.2f}\n"
                f"Prob subida: {prob*100:.1f}%\n"
                f"Timeframe: {INTERVAL}\n"
                f"Vela: {candle_time}"
            )
            send_telegram(msg)

# ================= EJECUCIÓN =================

if __name__ == "__main__":
    run()
