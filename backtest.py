# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
SIGNALS_CSV = os.path.join(BASE_DIR, "data", "signals.csv")

def main():
    if not os.path.exists(SIGNALS_CSV):
        print("No hay historial todavía.")
        print("Deja correr el bot varias horas para generar data/signals.csv")
        return

    df = pd.read_csv(SIGNALS_CSV)
    df["ts"] = pd.to_datetime(df["ts"])
    df["candle_time"] = pd.to_datetime(df["candle_time"])

    print("\n=== BACKTEST SIMPLE (aproximado) ===\n")

    for symbol in df["symbol"].unique():
        d = df[df["symbol"] == symbol].sort_values("candle_time").reset_index(drop=True)

        d["signal"] = d["prob_up"] >= d["threshold"]
        d["next_price"] = d["price"].shift(-1)
        d = d.dropna()

        trades = d[d["signal"]]

        if trades.empty:
            print(f"{symbol}: no hay operaciones aún")
            continue

        trades["pnl"] = (trades["next_price"] - trades["price"]) / trades["price"]

        win_rate = (trades["pnl"] > 0).mean() * 100
        avg_pnl = trades["pnl"].mean() * 100
        total_pnl = trades["pnl"].sum() * 100

        print(f"{symbol}")
        print(f"  Operaciones: {len(trades)}")
        print(f"  Win rate:    {win_rate:.1f}%")
        print(f"  Avg PnL:     {avg_pnl:.3f}%")
        print(f"  Total PnL:   {total_pnl:.2f}%\n")

if __name__ == "__main__":
    main()
