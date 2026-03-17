import os, json, glob
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
SNAPS = os.path.join(BASE, "snapshots", "paper_state_*.json")
LOCK = os.path.join(BASE, "risk_lock.json")

MAX_DD = 0.10  # 10% drawdown

def cap(d): 
    return float(d.get("capital", 0.0))

def main():
    files = sorted(glob.glob(SNAPS))
    if len(files) < 2:
        print("No hay suficientes snapshots.")
        return

    vals = []
    for fp in files:
        with open(fp, "r", encoding="utf-8-sig") as f:
            d = json.load(f)
        vals.append(cap(d))

    peak = vals[0]
    dd_now = 0.0
    for v in vals:
        peak = max(peak, v)
        dd_now = (v - peak) / peak if peak > 0 else 0.0

    locked = (-dd_now) >= MAX_DD
    payload = {
        "locked": bool(locked),
        "max_dd": MAX_DD,
        "dd_now": dd_now,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }

    with open(LOCK, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("OK -> risk_lock.json", payload)

if __name__ == "__main__":
    main()
