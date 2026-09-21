#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Trading Analysis — Backend API untuk Custom GPT (ChatGPT Actions)
=====================================================================
Server ini adalah backend untuk dipakai oleh sebuah Custom GPT lewat fitur
"Actions". ChatGPT memanggil endpoint ini (server-to-server, HTTPS), mengambil
hasil analisis kuantitatif (Entry/SL/TP/Confidence), lalu MENARASIKAN-nya
dalam bahasa natural kepada user.

>>> Tidak ada eksekusi order. Tool ini ANALYSIS-ONLY. <<<

ENDPOINTS:
  GET /health            -> status check
  GET /analyze           -> analisis 1 simbol + 1 timeframe (M1, M5, M15, M30, H1, H4, D1)
  GET /analyze_multi     -> multi-timeframe confluence (default M15, H1, H4;
                            bisa diubah lewat parameter `timeframes`)

Deploy:
  - Render / Railway / Fly.io (gratis, HTTPS otomatis).
  - Set command: uvicorn server:app --host 0.0.0.0 --port $PORT
  - Salin URL https://<app>.onrender.com ke Custom GPT -> Actions.

CHANGELOG (perbaikan):
  1. FIX KeyError 'tbb': kolom taker-buy volume (tbb) tidak lagi dibuang.
     Kolom ini dibutuhkan engine.analyze() untuk Order Flow Delta / CVD.
  2. Semua timeframe (M1, M5, M15, M30, H1, H4, D1) dipetakan langsung di
     server (TF_MAP), tidak bergantung pada subset yang ada di engine.py.
  3. Kode parsing klines digabung ke satu fungsi (_klines_to_df) supaya
     jalur host utama dan jalur proxy tidak bisa berbeda lagi.
  4. /analyze_multi menerima parameter `timeframes` (mis. "M1,M5,M15").
"""
import os
import math
from datetime import datetime, timezone

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

# Reuse engine analisis yang sudah teruji (skor, indikator, setup builder).
# Catatan: timeframe di-parse oleh server ini sendiri (lihat TF_MAP).
from engine import (
    resolve_symbol, analyze, build_setup, fetch_spot,
)

# =====================================================================
# TIMEFRAME MAP (MT4/MT5 -> interval Binance)
# =====================================================================
TF_MAP = {
    "M1":  ("1m",  "M1"),
    "M5":  ("5m",  "M5"),
    "M15": ("15m", "M15"),
    "M30": ("30m", "M30"),
    "H1":  ("1h",  "H1"),
    "H4":  ("4h",  "H4"),
    "D1":  ("1d",  "D1"),
}

# Alias yang ramah user / GPT (semua di-uppercase sebelum dicek).
# Sengaja TIDAK memetakan "1M" karena di Binance "1M" = 1 bulan, bukan 1 menit.
TF_ALIASES = {
    "1": "M1", "5": "M5", "15": "M15", "30": "M30",
    "60": "H1", "240": "H4", "1440": "D1",
    "1MIN": "M1", "5MIN": "M5", "15MIN": "M15", "30MIN": "M30",
    "1H": "H1", "4H": "H4", "1D": "D1", "D": "D1",
    "60M": "H1", "240M": "H4",
}


def _parse_tf(timeframe: str):
    """Kembalikan (interval_binance, label). Raise ValueError bila tidak dikenal."""
    key = (timeframe or "").strip().upper()
    key = TF_ALIASES.get(key, key)
    if key not in TF_MAP:
        raise ValueError(
            f"Timeframe '{timeframe}' tidak dikenal. "
            f"Gunakan salah satu: {', '.join(TF_MAP.keys())}."
        )
    return TF_MAP[key]


# =====================================================================
# DATA LAYER (server-side, multi-host + proxy fallback)
# Binance bisa di-geo-block di region tertentu. Karena server dipilih oleh
# kita (bukan browser user), kita tetap pakai fallback agar andal.
# =====================================================================
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api-gcp.binance.com",
]
CORS_PROXY = "https://corsproxy.io/?url="

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "qav", "trades", "tbb", "tbq", "ignore"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume",
                "qav", "trades", "tbb", "tbq"]


def _klines_to_df(raw) -> pd.DataFrame:
    """
    Ubah respons klines Binance menjadi DataFrame.

    PENTING: kolom tbb (taker buy base volume) dan tbq WAJIB dipertahankan,
    karena engine.analyze() memakainya untuk Order Flow Delta / CVD.
    """
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df = df.drop(columns=["ignore"])
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


def _fetch_klines_robust(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """Ambil candlestick dari Binance dengan fallback multi-host + CORS proxy."""
    path = f"/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    last_err = None
    for host in BINANCE_HOSTS:
        url = host + path
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return _klines_to_df(r.json())
            last_err = f"{host} HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{host} {type(e).__name__}"
    # fallback CORS proxy
    try:
        full = "https://api.binance.com" + path
        r = requests.get(CORS_PROXY + requests.utils.quote(full, safe=""), timeout=20)
        if r.ok:
            return _klines_to_df(r.json())
        last_err = f"{last_err or ''} | proxy HTTP {r.status_code}"
    except Exception as e:
        last_err = f"{last_err or ''} | proxy {type(e).__name__}"
    raise RuntimeError(f"Tidak bisa ambil data {symbol} dari semua sumber. ({last_err})")


# =====================================================================
# RESPONSE BUILDERS
# =====================================================================
def _f(x):
    """Konversi ke float Python biasa (JSON-safe), None bila NaN."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def build_response(symbol: str, timeframe: str, rr: float = 2.0) -> dict:
    display, bsym, is_alias = resolve_symbol(symbol)
    bint, tf_label = _parse_tf(timeframe)
    df = _fetch_klines_robust(bsym, bint, limit=300)
    if df.empty:
        raise RuntimeError(f"Data kosong untuk {symbol}")
    a = analyze(df)
    s = build_setup(a, rr=rr)
    spot_price, spot_label = fetch_spot(bsym)

    spot_spread_pct = None
    if spot_price:
        spot_spread_pct = (a["price"] - spot_price) / spot_price * 100

    risk = _f(s["risk"])
    rr_actual = None
    if risk and risk > 0:
        rr_actual = abs(_f(s["tp2"]) - _f(s["entry"])) / risk

    return {
        "symbol": display,
        "binance_symbol": bsym,
        "via_alias": is_alias,
        "timeframe": tf_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": s["direction"],
        "confidence_pct": round(_f(s["conf"]) or 0, 1),
        "score": int(a["score"]),
        "price": _f(a["price"]),
        "spot_reference": {
            "price": _f(spot_price),
            "label": spot_label,
            "proxy_spread_pct": round(spot_spread_pct, 3) if spot_spread_pct is not None else None,
        },
        "setup": {
            "entry": _f(s["entry"]) if s["direction"] != "NEUTRAL" else None,
            "stop_loss": _f(s["sl"]) if s["direction"] != "NEUTRAL" else None,
            "tp1": _f(s["tp1"]) if s["direction"] != "NEUTRAL" else None,
            "tp2": _f(s["tp2"]) if s["direction"] != "NEUTRAL" else None,
            "tp3": _f(s["tp3"]) if s["direction"] != "NEUTRAL" else None,
            "risk_per_unit": risk if s["direction"] != "NEUTRAL" else None,
            "risk_reward": round(rr_actual, 2) if rr_actual is not None else None,
            "sl_distance": _f(s["sl_dist"]) if s["direction"] != "NEUTRAL" else None,
            "note": (
                "Tidak ada setup jelas (NEUTRAL) — tunggu konfirmasi arah. "
                "Skor berada di rentang -20..+20 (tidak cukup kuat untuk LONG/SHORT)."
                if s["direction"] == "NEUTRAL" else None
            ),
        },
        "indicators": {
            "rsi14": _f(a["rsi"]),
            "ema50": _f(a["ema50"]),
            "ema200": _f(a["ema200"]),
            "atr14": _f(a["atr"]),
            "bollinger_upper": _f(a["bb_u"]),
            "bollinger_mid": _f(a["bb_m"]),
            "bollinger_lower": _f(a["bb_l"]),
            "swing_high_20": _f(a["swing_high"]),
            "swing_low_20": _f(a["swing_low"]),
        },
        "reasons": a["reasons"],
        "disclaimer": "Analysis only. Not financial advice. Slippage & fees not included. Execute manually.",
    }


# =====================================================================
# FASTAPI APP
# =====================================================================
app = FastAPI(
    title="AI Trading Analysis API",
    version="1.0.1",
    description=(
        "Backend analisis trading (analysis-only, no auto-execution). "
        "Menghitung Entry / Stop Loss / Take Profit / Confidence dari "
        "indikator EMA, RSI, MACD, Bollinger, ATR. Mendukung crypto "
        "(BTCUSDT, ETHUSDT, SOLUSDT, ...) dan emas (XAUUSD/XAU/GOLD). "
        "Timeframe MT4/MT5: M1, M5, M15, M30, H1, H4, D1."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health check", description="Cek server hidup.")
def health():
    return {
        "status": "ok",
        "service": "ai-trading-analysis",
        "timeframes": list(TF_MAP.keys()),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/analyze",
    summary="Analisa satu simbol & timeframe",
    description=(
        "Berikan analisis trading lengkap (arah LONG/SHORT/NEUTRAL, Entry, "
        "Stop Loss, TP1/TP2/TP3, Risk:Reward, Confidence, indikator, alasan). "
        "Parameter: symbol (mis. BTCUSDT atau XAUUSD), timeframe (M1/M5/M15/M30/H1/H4/D1), "
        "rr (target risk:reward, default 2). Output untuk ChatGPT menarasikan."
    ),
)
def analyze_endpoint(
    symbol: str = Query("XAUUSD", description="Simbol: BTCUSDT, ETHUSDT, SOLUSDT, atau XAUUSD/XAU/GOLD (emas)."),
    timeframe: str = Query("H1", description="Timeframe MT4/MT5: M1, M5, M15, M30, H1, H4, D1."),
    rr: float = Query(2.0, ge=0.5, le=10, description="Target Risk:Reward (default 2)."),
):
    try:
        return build_response(symbol, timeframe, rr=rr)
    except ValueError as e:
        return {"error": "bad_request", "message": str(e)}
    except Exception as e:
        return {"error": "server_error", "message": f"{type(e).__name__}: {e}"}


@app.get(
    "/analyze_multi",
    summary="Analisa multi-timeframe (confluence)",
    description=(
        "Jalankan analisis pada beberapa timeframe sekaligus dan hitung "
        "confluence (kesepakatan arah). Default: M15, H1, H4. Ubah lewat "
        "parameter `timeframes` (dipisah koma, mis. 'M1,M5,M15' untuk scalping "
        "atau 'M30,H1,H4'). Maksimal 6 timeframe. Bila semua timeframe searah, "
        "setup jauh lebih kuat. Berguna untuk konfirmasi sebelum entry manual."
    ),
)
def analyze_multi_endpoint(
    symbol: str = Query("BTCUSDT", description="Simbol, mis. BTCUSDT atau XAUUSD."),
    rr: float = Query(2.0, ge=0.5, le=10, description="Target Risk:Reward (default 2)."),
    timeframes: str = Query(
        "M15,H1,H4",
        description="Daftar timeframe dipisah koma, mis. 'M1,M5,M15'. Pilihan: M1, M5, M15, M30, H1, H4, D1.",
    ),
):
    # Parse & validasi daftar timeframe (buang duplikat, jaga urutan)
    frames = []
    try:
        for raw in timeframes.split(","):
            if not raw.strip():
                continue
            label = _parse_tf(raw)[1]
            if label not in frames:
                frames.append(label)
    except ValueError as e:
        return {"error": "bad_request", "message": str(e)}
    if not frames:
        return {"error": "bad_request", "message": "Parameter timeframes kosong."}
    if len(frames) > 6:
        return {"error": "bad_request", "message": "Maksimal 6 timeframe per permintaan."}

    results = []
    dirs = []
    for tf in frames:
        try:
            r = build_response(symbol, tf, rr=rr)
            results.append(r)
            dirs.append(r["direction"])
        except Exception as e:
            results.append({"timeframe": tf, "error": f"{type(e).__name__}: {e}"})
            dirs.append("ERROR")

    # confluence (dihitung terhadap jumlah timeframe yang diminta)
    n = len(frames)
    longs = dirs.count("LONG")
    shorts = dirs.count("SHORT")
    if n == 1:
        confluence = f"SINGLE TIMEFRAME ({dirs[0]}) — tidak ada confluence"
    elif longs == n:
        confluence = "STRONG LONG"
    elif shorts == n:
        confluence = "STRONG SHORT"
    elif longs > n / 2:
        confluence = f"WEAK LONG ({longs}/{n} searah)"
    elif shorts > n / 2:
        confluence = f"WEAK SHORT ({shorts}/{n} searah)"
    else:
        confluence = "NO CONFLUENCE (arah campur)"
    return {
        "symbol": symbol.upper(),
        "confluence": confluence,
        "directions": dict(zip(frames, dirs)),
        "analyses": results,
        "disclaimer": "Analysis only. Not financial advice.",
    }


# Override OpenAPI agar memuat server URL (Custom GPT butuh field servers).
def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="AI Trading Analysis API",
        version="1.0.1",
        description=(
            "Analisis trading otomatis (analysis-only, TIDAK ada eksekusi order). "
            "Mengembalikan arah trade, Entry, Stop Loss, Take Profit (TP1/TP2/TP3), "
            "Risk:Reward, Confidence %, dan alasan berbasis indikator "
            "(EMA, RSI, MACD, Bollinger, ATR). Timeframe: M1, M5, M15, M30, H1, H4, D1. "
            "Gunakan action 'analyze' untuk satu timeframe, atau 'analyze_multi' "
            "untuk confluence beberapa timeframe (default M15+H1+H4)."
        ),
        routes=app.routes,
    )
    base = os.environ.get("BASE_URL", "").rstrip("/")
    if base:
        schema["servers"] = [{"url": base}]
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
