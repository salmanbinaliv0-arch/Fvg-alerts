"""
FVG alerts -> Telegram. Python port of the alert logic in
"FVG Volume Profile [ChartPrime]" (SETUP / ENTRY / TP_HIT / SL_HIT / CANCELLED).

The volume profile is only visual in the Pine Script, so it is not needed here.
Runs statelessly: replays recent candles, then sends only events on bars newer
than the last one already alerted (kept in state.json).

Usage:
    python fvg_alerts.py          # normal run
    python fvg_alerts.py --test   # sends a test Telegram message
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

# ---- Settings (override with environment variables) ----
SYMBOL = os.getenv("SYMBOL", "XAU/USD")     # spot gold (Twelve Data)
INTERVAL = os.getenv("INTERVAL", "15min")   # 1min, 5min, 15min, 30min, 1h
API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
GAP_FILTER = float(os.getenv("GAP_FILTER", "0.5"))
RR = float(os.getenv("RR", "2.0"))
STDEV_LEN = 200
MAX_FVGS = 10
STATE_FILE = "state.json"

TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram secrets missing, printing instead:\n" + text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )
    r.raise_for_status()


def load_data():
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": SYMBOL, "interval": INTERVAL,
                "outputsize": 600, "timezone": "UTC",
                "apikey": API_KEY},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(f"Twelve Data error: {j.get('message', j)}")
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna()
    # keep only fully closed bars (Pine alerts fire at bar close)
    now = pd.Timestamp.now(tz="UTC")
    delta = pd.Timedelta(INTERVAL)
    return df[df.index + delta <= now]


def replay(df):
    H = df["High"].to_numpy(float)
    L = df["Low"].to_numpy(float)
    bear_raw = pd.Series(np.r_[np.nan, np.nan, L[:-2]] - H)
    bull_raw = pd.Series(L - np.r_[np.nan, np.nan, H[:-2]])
    # ta.stdev is the population stdev (ddof=0)
    bear_sz = (bear_raw / bear_raw.rolling(STDEV_LEN).std(ddof=0)).to_numpy()
    bull_sz = (bull_raw / bull_raw.rolling(STDEV_LEN).std(ddof=0)).to_numpy()

    events, fvgs = [], []

    def add(i, event, action, f):
        events.append({
            "ts": df.index[i], "event": event, "action": action,
            "entry": f["entry"], "sl": f["sl"], "tp": f["tp"],
        })

    for i in range(2, len(df)):
        bull = L[i] > H[i-2] and H[i-1] > H[i-2] and bull_sz[i] > GAP_FILTER
        bear = H[i] < L[i-2] and L[i-1] < L[i-2] and bear_sz[i] > GAP_FILTER

        if bull:
            entry, sl = L[i], H[i-2]
            f = dict(bull=True, entry=entry, sl=sl,
                     tp=entry + (entry - sl) * RR,
                     created=i, entered=False, tp_alerted=False)
            fvgs.append(f)
            add(i, "SETUP", "BUY", f)
        if bear:
            entry, sl = H[i], L[i-2]
            f = dict(bull=False, entry=entry, sl=sl,
                     tp=entry - (sl - entry) * RR,
                     created=i, entered=False, tp_alerted=False)
            fvgs.append(f)
            add(i, "SETUP", "SELL", f)

        if len(fvgs) > MAX_FVGS:
            fvgs.pop(0)  # oldest dropped silently, like the Pine script

        for f in list(fvgs):
            if f["bull"]:
                if i > f["created"] and not f["entered"] and L[i] <= f["entry"]:
                    f["entered"] = True
                    add(i, "ENTRY", "BUY", f)
                if f["entered"] and not f["tp_alerted"] and H[i] >= f["tp"]:
                    f["tp_alerted"] = True
                    add(i, "TP_HIT", "BUY", f)
                if L[i] < f["sl"]:
                    add(i, "SL_HIT" if f["entered"] else "CANCELLED", "BUY", f)
                    fvgs.remove(f)
            else:
                if i > f["created"] and not f["entered"] and H[i] >= f["entry"]:
                    f["entered"] = True
                    add(i, "ENTRY", "SELL", f)
                if f["entered"] and not f["tp_alerted"] and L[i] <= f["tp"]:
                    f["tp_alerted"] = True
                    add(i, "TP_HIT", "SELL", f)
                if H[i] > f["sl"]:
                    add(i, "SL_HIT" if f["entered"] else "CANCELLED", "SELL", f)
                    fvgs.remove(f)
    return events


def format_msg(e):
    icon = "🟢" if e["action"] == "BUY" else "🔴"
    titles = {
        "SETUP": f"{icon} {e['action']} SETUP",
        "ENTRY": f"{icon} ENTRY FILLED ({e['action']})",
        "TP_HIT": f"✅ TAKE PROFIT HIT ({e['action']})",
        "SL_HIT": f"❌ STOP LOSS HIT ({e['action']})",
        "CANCELLED": f"⚪ SETUP CANCELLED ({e['action']})",
    }
    t = e["ts"].strftime("%Y-%m-%d %H:%M UTC")
    return (f"{titles[e['event']]}\n{SYMBOL} {INTERVAL}\n"
            f"Entry: {e['entry']:.2f}\nStop Loss: {e['sl']:.2f}\n"
            f"Take Profit: {e['tp']:.2f}\nRR: 1:{RR:g}\nBar: {t}")


def main():
    if "--test" in sys.argv:
        send_telegram("✅ FVG alert bot is connected.")
        return

    df = load_data()
    if len(df) < STDEV_LEN + 5:
        print("Not enough data, skipping.")
        return
    events = replay(df)
    latest = df.index[-1].isoformat()

    if os.path.exists(STATE_FILE):
        last = pd.Timestamp(json.load(open(STATE_FILE))["last_ts"])
        new = [e for e in events if e["ts"] > last]
        for e in new:
            send_telegram(format_msg(e))
        print(f"Sent {len(new)} alert(s).")
    else:
        # first run: don't spam old history
        send_telegram(f"✅ FVG bot started on {SYMBOL} {INTERVAL}. Alerts from now on.")
        print("First run, state initialised.")

    json.dump({"last_ts": latest}, open(STATE_FILE, "w"))


if __name__ == "__main__":
    main()

