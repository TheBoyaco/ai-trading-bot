# -*- coding: utf-8 -*-
"""
Brain AutoTune (PAPER/TESTNET) - ULTRA ROBUST
- Prioriza trades_testnet.csv si existe.
- Si no, usa paper_trades.csv.
- Para PAPER usa parser manual con csv.reader (tolera filas con diferente # de columnas).
- PAPER: optimiza SOLO prob_min (porque no hay RVOL/buy_ratio).
- Siempre hace backup del config antes de cambiar.
"""

import os, json, shutil, csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests


BASE_DIR = r"C:\bot_ia_trading"
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def utc_now():
    return datetime.now(timezone.utc)


def send_telegram(token: str, chat_id: str, msg: str):
    if not token or not chat_id or "TU_" in token:
        print(msg)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=25)
    except Exception as e:
        print("Telegram error:", e)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default


def pick_trades_file() -> str:
    candidates = [
        os.path.join(DATA_DIR, "trades_testnet.csv"),
        os.path.join(DATA_DIR, "paper_trades.csv"),
        os.path.join(DATA_DIR, "trades.csv"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 20:
            return p
    return candidates[0]


def profit_factor(pnls: pd.Series) -> float:
    pnls = pnls.astype(float)
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def score_set(df_sel: pd.DataFrame):
    n = len(df_sel)
    if n == 0:
        return None
    pnls = df_sel["pnl_usdt"].astype(float)
    total = float(pnls.sum())
    wr = float((pnls > 0).mean()) if n else 0.0
    pf = profit_factor(pnls)
    avg = float(pnls.mean()) if n else 0.0
    return {"n": n, "total": total, "wr": wr, "pf": pf, "avg": avg}


# -------------------------
# PAPER parser (manual)
# -------------------------
def read_paper_trades_manual(path: str) -> pd.DataFrame:
    """
    paper_trades.csv:
      ts_utc,symbol,interval,action,price,prob_up,capital_before,capital_after,candle_time
    A veces EXIT trae columnas extra. No importa: tomamos índices fijos 0..7.
    Usamos SOLO EXIT para calcular pnl_usdt = capital_after - capital_before.
    """
    rows = []
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time_utc","symbol","pnl_usdt","entry_prob"])

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)  # saltar encabezado
        for row in r:
            if not row or len(row) < 8:
                continue
            action = str(row[3]).strip().upper()
            if action != "EXIT":
                continue

            ts = str(row[0]).strip()
            sym = str(row[1]).strip()

            prob = safe_float(row[5], None)
            cap_before = safe_float(row[6], None)
            cap_after  = safe_float(row[7], None)

            if prob is None or cap_before is None or cap_after is None:
                continue

            pnl = cap_after - cap_before
            rows.append({
                "time_utc": ts,
                "symbol": sym,
                "pnl_usdt": float(pnl),
                "entry_prob": float(prob),
            })

    df = pd.DataFrame(rows)
    return df


# -------------------------
# TESTNET reader (simple)
# -------------------------
def read_testnet_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["pnl_usdt", "entry_prob"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["pnl_usdt", "entry_prob"])
    return df


# -------------------------
# PAPER tuning: prob_min only
# -------------------------
def choose_best_prob_only(df: pd.DataFrame, base_cfg: dict):
    prob0 = safe_float(base_cfg.get("prob_min", 0.55), 0.55)
    bounds = (0.50, 0.70)

    prob_grid = sorted(set([round(x, 3) for x in [
        prob0 - 0.05, prob0 - 0.03, prob0 - 0.02, prob0 - 0.01,
        prob0, prob0 + 0.01, prob0 + 0.02, prob0 + 0.03, prob0 + 0.05
    ] if bounds[0] <= x <= bounds[1]]))

    min_trades = 3  # PAPER al inicio: con 3 ya empezamos a aprender suave

    base_sel = df[df["entry_prob"] >= prob0]
    base_m = score_set(base_sel) or {"n": 0, "total": 0.0, "wr": 0.0, "pf": 0.0, "avg": 0.0}

    best = None
    best_m = None
    best_p = prob0

    for p in prob_grid:
        sel = df[df["entry_prob"] >= p]
        m = score_set(sel)
        if not m or m["n"] < min_trades:
            continue
        obj = (m["total"] * 1.0) + (m["pf"] * 0.8) + (m["wr"] * 0.3) - (m["n"] * 0.01)
        if (best is None) or (obj > best):
            best = obj
            best_m = m
            best_p = p

    if best_m is None:
        return None, base_m, prob0

    improves_total = best_m["total"] >= (base_m["total"] + 0.01)
    improves_pf = best_m["pf"] >= max(1.05, base_m["pf"] * 0.98)

    if improves_total and improves_pf and best_p != prob0:
        return best_p, base_m, best_m

    return None, base_m, prob0


def main():
    if not os.path.exists(CONFIG_PATH):
        print("No existe config.json en C:\\bot_ia_trading")
        return

    cfg = load_json(CONFIG_PATH)
    token = cfg.get("telegram_token", "")
    chat_id = cfg.get("telegram_chat_id", "")

    trades_path = pick_trades_file()
    source = os.path.basename(trades_path).lower()

    if not os.path.exists(trades_path) or os.path.getsize(trades_path) < 20:
        msg = "🧠 Brain: No hay trades todavía. Deja el bot operar y luego intento de nuevo."
        send_telegram(token, chat_id, msg)
        print(msg)
        return

    if source == "paper_trades.csv":
        df = read_paper_trades_manual(trades_path)
        mode = "PAPER"
    else:
        df = read_testnet_trades(trades_path)
        mode = "TESTNET"

    if len(df) < 1:
        msg = f"🧠 Brain: No pude extraer trades útiles.\nFuente: {source}\nTip: en PAPER solo cuento EXIT."
        send_telegram(token, chat_id, msg)
        print(msg)
        return

    # Ventana: últimos 7 días si parsea time_utc
    df_use = df.copy()
    window_label = "todo el historial"
    try:
        dts = pd.to_datetime(df_use["time_utc"], errors="coerce", utc=True)
        df_use = df_use.assign(dt=dts).dropna(subset=["dt"])
        if len(df_use) >= 3:
            cut = utc_now() - timedelta(days=7)
            df_recent = df_use[df_use["dt"] >= cut].copy()
            if len(df_recent) >= 1:
                df_use = df_recent
                window_label = "últimos 7 días"
    except:
        pass

    total_trades = len(df_use)
    total_pnl = float(df_use["pnl_usdt"].sum())
    wr_total = float((df_use["pnl_usdt"] > 0).mean()) if total_trades else 0.0
    pf_total = profit_factor(df_use["pnl_usdt"])

    header = (
        f"🧠 Brain AutoTune ({mode})\n"
        f"Fuente: {source}\n"
        f"Ventana: {window_label}\n"
        f"Trades: {total_trades} | PnL: {total_pnl:+.2f} | WR: {wr_total*100:.1f}% | PF: {pf_total:.2f}\n"
    )

    if mode == "PAPER":
        best_p, base_m, best_m = choose_best_prob_only(df_use, cfg)
        if best_p is None:
            msg = header + f"✅ No aplico cambios hoy.\nConfig actual: prob_min={cfg.get('prob_min')}\n"
            send_telegram(token, chat_id, msg)
            print(msg)
            return

        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BASE_DIR, f"config_backup_{ts}.json")
        shutil.copy2(CONFIG_PATH, backup_path)

        old = cfg.get("prob_min")
        cfg["prob_min"] = float(best_p)
        save_json(CONFIG_PATH, cfg)

        msg = (
            header +
            "✅ APLIQUÉ AJUSTE (PAPER)\n"
            f"ANTES: prob_min={old}\n"
            f"AHORA:  prob_min={cfg['prob_min']}\n"
            f"Mejora subset: trades={best_m['n']} | PnL={best_m['total']:+.2f} | WR={best_m['wr']*100:.1f}% | PF={best_m['pf']:.2f}\n"
            f"Backup: {os.path.basename(backup_path)}\n"
        )
        send_telegram(token, chat_id, msg)
        print(msg)
        return

    msg = header + "ℹ️ TESTNET detectado. (Cuando existan trades_testnet.csv completos, ampliamos a RVOL/buy_ratio.)"
    send_telegram(token, chat_id, msg)
    print(msg)


if __name__ == "__main__":
    main()
