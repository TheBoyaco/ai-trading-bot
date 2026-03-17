import os, json
from datetime import datetime, timezone

BASE = r"C:\bot_ia_trading"
HUB = os.path.join(BASE, r"data\brain\hub.json")
STATE = os.path.join(BASE, r"data\paper_state.json")
LOG_MAIN = os.path.join(BASE, "paper_logs.txt")
LOG_TASK = os.path.join(BASE, "paper_task.log")

MAX_AGE_HUB = 180   # segundos (3 min)
MAX_AGE_PAPER = 180 # segundos (3 min)
MAX_AGE_STATE = 300 # segundos (5 min)

def now_utc():
    return datetime.now(timezone.utc)

def file_age_seconds(path: str):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)  # epoch
    age = now_utc().timestamp() - mtime
    return int(age)

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}
    except Exception:
        return {}

def pick_live_log():
    a = file_age_seconds(LOG_MAIN)
    b = file_age_seconds(LOG_TASK)

    # Elige el mÃƒÆ’Ã‚Â¡s reciente (menor age), ignorando None
    candidates = []
    if a is not None: candidates.append((a, LOG_MAIN))
    if b is not None: candidates.append((b, LOG_TASK))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]  # path, age

def parse_state_updated_utc(state_json):
    try:
        s = state_json.get("updated_utc")
        if not s:
            return None
        # Ej: "2025-12-27T15:29:04+00:00"
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int((now_utc() - dt).total_seconds())
    except Exception:
        return None

def ok_line(name, ok, detail):
    status = "OK" if ok else f"NO ACTUALIZA ({detail}s)"
    print(f"{name:<12}: {status}")

def main():
    print("\n==============================")
    print("ESTADO RED NEURONAL TRADING")
    print("==============================\n")

    # HUB
    hub_age = file_age_seconds(HUB)
    hub_ok = (hub_age is not None and hub_age <= MAX_AGE_HUB)
    ok_line("HUB", hub_ok, hub_age if hub_age is not None else 999999)

    # PAPER LOG (auto)
    live_log, live_age = pick_live_log()
    paper_ok = (live_log is not None and live_age <= MAX_AGE_PAPER)
    if live_log is None:
        print("PAPER BOT    : NO HAY LOG (paper_logs.txt / paper_task.log)")
    else:
        src = os.path.basename(live_log)
        status = "OK" if paper_ok else f"NO ACTUALIZA ({live_age}s)"
        print(f"PAPER BOT    : {status}  [log={src}]")

    # STATE
    state = read_json(STATE)
    if not state:
        print("ESTADO POS   : NO EXISTE paper_state.json")
        state_ok = False
        state_age = None
    else:
        state_age = parse_state_updated_utc(state)
        state_ok = (state_age is not None and state_age <= MAX_AGE_STATE)
        ok_line("ESTADO POS", state_ok, state_age if state_age is not None else 999999)

    print("\n------------------------------")
    if hub_ok and paper_ok and state_ok:
        print("TODO OK ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â RED ACTIVA Y TRABAJANDO")
    else:
        print("ATENCION ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â HAY PROBLEMAS ARRIBA")
    print("------------------------------\n")

if __name__ == "__main__":
    main()

