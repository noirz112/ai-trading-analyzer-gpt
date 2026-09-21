#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Trading Analysis Tool (Analysis-Only / Manual Execution)
============================================================
Tool ini bekerja sebagai ANALIS — bukan eksekutor. Outputnya:
  - Arah (LONG / SHORT / NEUTRAL)
  - Entry (zona)
  - Stop Loss (SL)
  - Take Profit (TP1, TP2, TP3) multi-level
  - Risk:Reward
  - Confidence (%)
  - Alasan (reasoning) berbasis indikator
  - Tingkat dukungan / resistensi terdekat

>>> SEMUA EKSEKUSI TETAP DILAKUKAN MANUAL OLEH USER. <<<

Sumber data  : Binance public API (TANPA API key).
Indikator    : EMA50/200, RSI(14), MACD, Bollinger Bands, ATR(14).
Skoring      : Multi-faktor (-100 ... +100). Model kuantitatif berbasis aturan
               (rule-based expert system). Bisa di-extend ke LLM (lihat bawah).

DUKUNGAN ASSET:
  - Crypto  : BTCUSDT, ETHUSDT, SOLUSDT, dll. (langsung dari Binance)
  - EMAS    : XAUUSD / XAU / GOLD  -> otomatis memakai PAXGUSDT
              (PAX Gold = token 1:1 emas fisik, melacak spot XAUUSD).
              Tool juga menampilkan cross-check harga spot emas dari API
              independen (gold-api.com) untuk transparansi tracking.

CARA PAKAI:
  python trading_analyzer.py BTCUSDT H4
  python trading_analyzer.py ETHUSDT H1 --rr 2
  python trading_analyzer.py XAUUSD D1 --chart        # emas harian
  python trading_analyzer.py GOLD M15 --chart
  python trading_analyzer.py BTCUSDT H4 --list BTCUSDT ETHUSDT XAUUSD

TIMEFRAME yang didukung (MT4/MT5 style, case-insensitive):
  M1, M5, M15, M30, H1, H4, D1
  (juga menerima format Binance: 1m, 5m, 15m, 30m, 1h, 4h, 1d)

DEPENDENCY: pandas, numpy, requests, matplotlib (sudah pre-installed).
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# Alias: nama akrab -> simbol Binance yang tersedia tanpa API key.
# PAXGUSDT = PAX Gold (token 1:1 emas fisik), melacak spot XAUUSD.
SYMBOL_ALIASES = {
    "XAUUSD": "PAXGUSDT",
    "XAU": "PAXGUSDT",
    "GOLD": "PAXGUSDT",
    "PAXG": "PAXGUSDT",
}

# Cross-check harga spot independen (optional, transparansi tracking proxy).
SPOT_CHECK = {
    "PAXGUSDT": ("https://api.gold-api.com/price/XAU", "XAUUSD spot"),
}

# Optional: integrasi LLM untuk narasi natural-language.
# Uncomment & isi API key bila ingin AI menarasikan hasil analisis.
# import os
# def llm_narrate(context_block):
#     headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
#     prompt = ("Anda analis trading. Berdasarkan konteks berikut, buat ringkasan "
#               "singkat mengapa setup ini layak/tidak layak, dan peringatan risiko.\n\n"
#               + context_block)
#     r = requests.post("https://api.openai.com/v1/chat/completions",
#                       headers=headers, json={"model":"gpt-4o-mini",
#                       "messages":[{"role":"user","content":prompt}]}, timeout=30)
#     return r.json()["choices"][0]["message"]["content"]


# =====================================================================
# 0. ALIAS RESOLVER + SPOT CROSS-CHECK
# =====================================================================
def resolve_symbol(symbol: str):
    """Kembalikan (display_name, binance_symbol, is_alias)."""
    key = symbol.upper().strip()
    if key in SYMBOL_ALIASES:
        return key, SYMBOL_ALIASES[key], True
    return key, key, False


def fetch_spot(binance_symbol: str):
    """Ambil harga spot referensi (mis. XAUUSD) bila ada. Kembalikan (price, label) atau (None, label)."""
    if binance_symbol not in SPOT_CHECK:
        return None, None
    url, label = SPOT_CHECK[binance_symbol]
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        price = float(data.get("price") or data.get("rate") or 0)
        return price, label
    except Exception:
        return None, label


# =====================================================================
# 0b. TIMEFRAME PARSER (MT4/MT5 style + Binance native)
# =====================================================================
# Memetakan input user -> (binance_interval, display_label)
TIMEFRAMES = {
    # MT4/MT5 style
    "M1":  ("1m",  "M1"),
    "M5":  ("5m",  "M5"),
    "M15": ("15m", "M15"),
    "M30": ("30m", "M30"),
    "H1":  ("1h",  "H1"),
    "H4":  ("4h",  "H4"),
    "D1":  ("1d",  "D1"),
    # Binance native (diterima juga)
    "1M":  ("1m",  "M1"),
    "5M":  ("5m",  "M5"),
    "15M": ("15m", "M15"),
    "30M": ("30m", "M30"),
    "1H":  ("1h",  "H1"),
    "4H":  ("4h",  "H4"),
    "1D":  ("1d",  "D1"),
}


def parse_timeframe(tf: str):
    """Kembalikan (binance_interval, display_label). Raise ValueError bila tidak dikenal."""
    key = tf.upper().strip()
    if key in TIMEFRAMES:
        return TIMEFRAMES[key]
    raise ValueError(
        f"Timeframe '{tf}' tidak dikenal. Pilihan yang valid: "
        f"M1, M5, M15, M30, H1, H4, D1  (atau 1m, 5m, 15m, 30m, 1h, 4h, 1d)"
    )


# =====================================================================
# 1. DATA LAYER
# =====================================================================
def fetch_klines(symbol: str, interval: str = "4h", limit: int = 300) -> pd.DataFrame:
    """Ambil candlestick (OHLCV) dari Binance public API. Tanpa API key."""
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=15)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


# =====================================================================
# 2. INDICATOR LAYER
# =====================================================================
def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))


def macd(s: pd.Series, f: int = 12, sl: int = 26, sig: int = 9):
    m = ema(s, f) - ema(s, sl)
    sigl = m.ewm(span=sig, adjust=False).mean()
    return m, sigl, m - sigl


def bollinger(s: pd.Series, p: int = 20, k: float = 2.0):
    mid = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return mid + k * sd, mid, mid - k * sd


def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(p).mean()


# =====================================================================
# 3. SCORING ENGINE (multi-faktor, -100 ... +100)
# =====================================================================
def analyze(df: pd.DataFrame) -> dict:
    c = df["close"]
    e50, e200 = ema(c, 50), ema(c, 200)
    r = rsi(c, 14)
    macd_l, macd_s, hist = macd(c)
    bb_u, bb_m, bb_l = bollinger(c)
    a = atr(df, 14)

    score = 0
    reasons = []

    # --- Trend (EMA) ---
    if e50.iloc[-1] > e200.iloc[-1]:
        score += 25
        reasons.append("EMA50 > EMA200  (uptrend)")
    else:
        score -= 25
        reasons.append("EMA50 < EMA200  (downtrend)")
    if c.iloc[-1] > e50.iloc[-1]:
        score += 15
        reasons.append("Harga di atas EMA50")
    else:
        score -= 15
        reasons.append("Harga di bawah EMA50")

    # --- Momentum (RSI) ---
    r_now = r.iloc[-1]
    if r_now < 30:
        score += 20
        reasons.append(f"RSI {r_now:.1f}  -> oversold (potensi rebound)")
    elif r_now > 70:
        score -= 20
        reasons.append(f"RSI {r_now:.1f}  -> overbought (potensi pullback)")
    else:
        reasons.append(f"RSI {r_now:.1f}  -> netral")

    # --- Konfirmasi (MACD) ---
    if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
        score += 20
        reasons.append("MACD bullish cross (histogram membalik ke +)")
    elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
        score -= 20
        reasons.append("MACD bearish cross (histogram membalik ke -)")
    elif hist.iloc[-1] > 0:
        score += 8
        reasons.append("MACD histogram positif (momentum bullish)")
    else:
        score -= 8
        reasons.append("MACD histogram negatif (momentum bearish)")

    # --- Volatilitas / extremes (Bollinger) ---
    if c.iloc[-1] < bb_l.iloc[-1]:
        score += 15
        reasons.append("Harga menembus LOWER Bollinger (oversold ekstrem)")
    elif c.iloc[-1] > bb_u.iloc[-1]:
        score -= 15
        reasons.append("Harga menembus UPPER Bollinger (overbought ekstrem)")
    else:
        reasons.append("Harga di dalam band Bollinger (normal)")

    # --- Support / Resistance (swing 20 bar) ---
    swing_high = df["high"].rolling(20).max().iloc[-1]
    swing_low = df["low"].rolling(20).min().iloc[-1]
    reasons.append(f"Resistance 20-bar: {swing_high:.4f}   |   Support 20-bar: {swing_low:.4f}")

    return dict(
        score=score, reasons=reasons,
        price=float(c.iloc[-1]), atr=float(a.iloc[-1]),
        rsi=float(r_now), ema50=float(e50.iloc[-1]), ema200=float(e200.iloc[-1]),
        bb_u=float(bb_u.iloc[-1]), bb_m=float(bb_m.iloc[-1]), bb_l=float(bb_l.iloc[-1]),
        swing_high=float(swing_high), swing_low=float(swing_low),
    )


# =====================================================================
# 4. SETUP BUILDER (Entry / SL / TP berbasis ATR)
# =====================================================================
def build_setup(a: dict, rr: float = 2.0, atr_mult: float = 1.5) -> dict:
    price = a["price"]
    atrv = a["atr"]
    sl_dist = atr_mult * atrv  # jarak SL dari entry, proporsional volatilitas

    if a["score"] > 20:
        direction = "LONG"
    elif a["score"] < -20:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # confidence: petakan |score| ke 50..90
    conf = min(90.0, 50.0 + abs(a["score"]))

    if direction == "LONG":
        entry = price
        sl = entry - sl_dist
        tp1 = entry + sl_dist * 1.0
        tp2 = entry + sl_dist * rr
        tp3 = entry + sl_dist * (rr * 1.5)
        risk = entry - sl
    elif direction == "SHORT":
        entry = price
        sl = entry + sl_dist
        tp1 = entry - sl_dist * 1.0
        tp2 = entry - sl_dist * rr
        tp3 = entry - sl_dist * (rr * 1.5)
        risk = sl - entry
    else:
        entry = price
        sl = tp1 = tp2 = tp3 = price
        risk = 0.0

    return dict(direction=direction, entry=entry, sl=sl,
                tp1=tp1, tp2=tp2, tp3=tp3, rr=rr, risk=risk,
                conf=conf, sl_dist=sl_dist)


# =====================================================================
# 5. REPORT PRINTER
# =====================================================================
def fmt(x, dp=4):
    return f"{x:.{dp}f}"


def print_report(symbol, binance_symbol, tf_label, bint, a, s, is_alias=False,
                 spot_price=None, spot_label=None, chart=False):
    line = "=" * 64
    print(f"\n{line}")
    if is_alias:
        print(f"  AI TRADING ANALYSIS  |  {symbol}  {tf_label}   (data via {binance_symbol})")
    else:
        print(f"  AI TRADING ANALYSIS  |  {symbol}  {tf_label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    (ANALYSIS-ONLY, no auto-trade)")
    print(line)

    arrow = {"LONG": "▲ LONG (beli)", "SHORT": "▼ SHORT (jual)",
             "NEUTRAL": "= NEUTRAL (tidak ada setup jelas)"}[s["direction"]]

    print(f"  Arah        : {arrow}")
    print(f"  Confidence  : {s['conf']:.0f}%   (skor multi-faktor: {a['score']:+.0f})")
    print(f"  Harga now   : {fmt(a['price'])}")
    # Cross-check spot referensi (mis. XAUUSD asli) bila tersedia
    if spot_price is not None:
        spread = a['price'] - spot_price
        spread_pct = (spread / spot_price * 100) if spot_price else 0
        print(f"  Spot ref    : {fmt(spot_price)}  ({spot_label})   "
              f"selisih proxy: {spread:+.2f} ({spread_pct:+.2f}%)")
    print(f"  ATR(14)     : {fmt(a['atr'])}   -> jarak SL: {fmt(s['sl_dist'])}")
    print("-" * 64)
    if s["direction"] != "NEUTRAL":
        print(f"  ENTRY       : {fmt(s['entry'])}")
        print(f"  STOP LOSS   : {fmt(s['sl'])}    (risk: {fmt(s['risk'])} /candle-unit)")
        print(f"  TP1 (1R)    : {fmt(s['tp1'])}")
        print(f"  TP2 ({s['rr']}R)  : {fmt(s['tp2'])}")
        print(f"  TP3 ({s['rr']*1.5:g}R) : {fmt(s['tp3'])}")
        # validasi R:R aktual
        if s["risk"] > 0:
            rr_actual = abs(s["tp2"] - s["entry"]) / s["risk"]
            print(f"  Risk:Reward : 1 : {rr_actual:.2f}")
    else:
        print("  Tidak ada setup jelas. Tunggu konfirmasi (skor di rentang -20..+20).")
    print("-" * 64)
    print("  KEY LEVELS")
    print(f"   EMA50  : {fmt(a['ema50'])}    EMA200 : {fmt(a['ema200'])}")
    print(f"   BB up  : {fmt(a['bb_u'])}    BB mid: {fmt(a['bb_m'])}    BB low: {fmt(a['bb_l'])}")
    print(f"   Resist : {fmt(a['swing_high'])}    Support: {fmt(a['swing_low'])}")
    print("-" * 64)
    print("  ALASAN (reasoning):")
    for r_ in a["reasons"]:
        print(f"   - {r_}")
    print("-" * 64)
    print("  CATATAN RISIKO:")
    print("   - Ini ANALISIS, bukan saran keuangan. DYOR (Do Your Own Research).")
    print("   - Slippage & fee tidak dihitung di sini -> sesuaikan saat eksekusi.")
    print("   - Pasar sangat volatil; SL bisa tersentuh karena spike (gap).")
    print("   - Confidence tinggi TIDAK menjamin profit. Kelola posisi & ukuran.")
    if is_alias:
        print(f"   - {symbol} memakai proxy {binance_symbol}: melacak spot, "
              f"bukan harga XAUUSD persis. Ada selisih kecil (lihat Spot ref).")
    print(line + "\n")

    if chart:
        try:
            plot_chart(symbol, binance_symbol, tf_label, bint, a, s)
        except Exception as e:
            print(f"  [chart gagal: {e}]")


# =====================================================================
# 6. CHART (opsional)
# =====================================================================
def plot_chart(symbol, binance_symbol, tf_label, bint, a, s, window=90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # re-fetch untuk plot (pakai binance_symbol + interval Binance, bukan alias/label)
    df = fetch_klines(binance_symbol, bint, limit=window)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["open_time"], df["close"], color="#1f77b4", linewidth=1.4, label="Close")

    # level-line entry/sl/tp
    if s["direction"] != "NEUTRAL":
        levels = [("Entry", s["entry"], "#2ca02c"),
                  ("SL", s["sl"], "#d62728"),
                  ("TP1", s["tp1"], "#ff7f0e"),
                  ("TP2", s["tp2"], "#9467bd"),
                  ("TP3", s["tp3"], "#8c564b")]
        for name, val, col in levels:
            ax.axhline(val, color=col, linestyle="--", linewidth=1.1, alpha=0.85)
            ax.text(df["open_time"].iloc[-1], val, f" {name} {val:.4f}",
                    color=col, fontsize=8, va="center", ha="left")

    # EMA
    ax.plot(df["open_time"], ema(df["close"], 50), color="orange",
            linewidth=1, alpha=0.7, label="EMA50")
    # EMA200 hanya bila data cukup; kalau kurang, gunakan periode maksimum yang tersedia
    avail = len(df) - 1
    ema200_period = 200 if avail >= 200 else max(50, avail)
    label200 = "EMA200" if ema200_period >= 200 else f"EMA{ema200_period}"
    ax.plot(df["open_time"], ema(df["close"], ema200_period),
            color="gray", linewidth=1, alpha=0.7, label=label200)

    title_extra = f"  (via {binance_symbol})" if symbol != binance_symbol else ""
    ax.set_title(f"{symbol}{title_extra} {tf_label}  -  {s['direction']}  conf {s['conf']:.0f}%")
    ax.set_xlabel("Waktu"); ax.set_ylabel("Harga")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    fname = f"chart_{symbol}_{tf_label}.png"
    plt.savefig(fname, dpi=110)
    plt.close()
    print(f"  [chart disimpan: {fname}]")


# =====================================================================
# 7. MAIN
# =====================================================================
def run(symbol, interval, rr=2.0, chart=False):
    display, bsym, is_alias = resolve_symbol(symbol)
    bint, tf_label = parse_timeframe(interval)
    df = fetch_klines(bsym, bint)
    a = analyze(df)
    s = build_setup(a, rr=rr)
    spot_price, spot_label = fetch_spot(bsym)
    print_report(display, bsym, tf_label, bint, a, s,
                 is_alias=is_alias, spot_price=spot_price,
                 spot_label=spot_label, chart=chart)
    return a, s


def main():
    ap = argparse.ArgumentParser(description="AI Trading Analysis (analysis-only)")
    ap.add_argument("symbol", help="Simbol: BTCUSDT/ETHUSDT, atau XAUUSD/XAU/GOLD (emas)")
    ap.add_argument("interval", help="Timeframe: M1, M5, M15, M30, H1, H4, D1")
    ap.add_argument("--rr", type=float, default=2.0, help="Target Risk:Reward (default 2)")
    ap.add_argument("--chart", action="store_true", help="Simpan chart PNG")
    ap.add_argument("--list", nargs="+", metavar="SYM", help="Analisis banyak simbol")
    args = ap.parse_args()

    if args.list:
        for sym in args.list:
            try:
                run(sym, args.interval, rr=args.rr, chart=args.chart)
            except Exception as e:
                print(f"[{sym}] gagal: {e}")
    else:
        try:
            run(args.symbol, args.interval, rr=args.rr, chart=args.chart)
        except ValueError as e:
            print(f"[error] {e}")
            sys.exit(2)
        except Exception as e:
            print(f"[{args.symbol}] gagal: {e}")


if __name__ == "__main__":
    main()
