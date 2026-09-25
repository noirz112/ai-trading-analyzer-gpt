#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Trading Analysis Tool v2 — Smart Money / Liquidity Based (Analysis-Only)
============================================================================
Versi ini MENGGANTI seluruh indikator klasik (EMA, RSI, MACD, Bollinger Bands,
ATR, Ichimoku) dengan pendekatan berbasis STRUKTUR & LIKUIDITAS:

UNTUK CRYPTO:
  1. ERL/IRL + Liquidity Pool           -> zona "magnet" (equal high/low, FVG)
  2. Liquidation Magnet (PROXY)         -> lihat catatan limitasi di bawah
  3. Order Flow + Volume Profile/POC + VWAP -> real-time (dari klines Binance)
  4. Struktur HTF (higher timeframe)    -> bias arah, BOS / CHoCH

UNTUK XAU (otomatis dipetakan ke PAXGUSDT, playbook HIBRIDA):
  1. Volume Profile/POC + VWAP          -> di-anchor ke sesi overlap London-NY
  2. Macro (DXY proxy, real yield, bias Fed) -> lihat catatan limitasi
  3. Reaksi candle + volume             -> pengganti order flow real-time
  4. Struktur HTF, S/R, sama seperti crypto
  5. Funding rate / OI / liquidation magnet dari PAXGUSDT -> HANYA jika
     kontrak futures PAXGUSDT tersedia di Binance (auto-detect; kalau tidak
     ada, bagian ini dilewati & bobotnya dialihkan ke faktor lain)
  6. COT report & COMEX OI              -> INFORMASIONAL SAJA (opsional,
     --cot), TIDAK dihitung ke skor karena datanya mingguan/tidak real-time

=============================================================================
CATATAN LIMITASI DATA (WAJIB DIBACA — supaya tidak salah kira ini data pasti):
=============================================================================
  - Funding rate & Open Interest  : REAL, gratis, tanpa API key
                                     (Binance Futures public endpoint).
  - Liquidation heatmap ASLI (spt Coinglass) : TIDAK tersedia gratis tanpa
    API key berbayar. Yang dihitung di sini adalah "Liquidation Magnet
    Proxy" -> heuristik dari liquidity pool (cluster stop-loss di equal
    high/low) + funding rate ekstrem + tren OI. INI BUKAN data notional
    likuidasi riil, hanya estimasi area rawan liquidity grab.
  - DXY               : proxy harian dari kurs referensi ECB (frankfurter.app,
                         gratis tanpa key) memakai formula USDX resmi.
                         Update harian, BUKAN tick real-time intraday.
  - Real yield & ekspektasi Fed : tidak ada sumber gratis tanpa API key yang
                         reliable -> disediakan sebagai PARAMETER MANUAL
                         opsional (--dxy-bias / --macro-note), bukan ditarik
                         otomatis. Jangan dianggap terisi otomatis.
  - COT report        : CFTC public API (gratis, tanpa key), tapi mingguan
                         (delayed beberapa hari) -> hanya ditampilkan sebagai
                         info tambahan (--cot), TIDAK mempengaruhi skor.

DEPENDENCY: pandas, numpy, requests, matplotlib (opsional untuk --chart).

CARA PAKAI:
  python engine.py BTCUSDT H4
  python engine.py ETHUSDT H1 --rr 2
  python engine.py XAUUSD D1 --chart --cot
  python engine.py GOLD H1 --dxy-bias bearish
  python engine.py BTCUSDT H4 --list BTCUSDT ETHUSDT XAUUSD
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# =============================================================================
# 0. ENDPOINTS & KONSTANTA
# =============================================================================
BINANCE_SPOT_KLINES = "https://data-api.binance.vision/api/v3/klines"
FAPI_BASE = "https://fapi.binance.com"
FAPI_KLINES = FAPI_BASE + "/fapi/v1/klines"
FAPI_PREMIUM_INDEX = FAPI_BASE + "/fapi/v1/premiumIndex"
FAPI_OPEN_INTEREST = FAPI_BASE + "/fapi/v1/openInterest"
FAPI_OI_HIST = FAPI_BASE + "/futures/data/openInterestHist"

FX_LATEST_URL = "https://api.frankfurter.app/latest"
FX_RANGE_URL = "https://api.frankfurter.app/{start}..{end}"
GOLD_SPOT_URL = "https://api.gold-api.com/price/XAU"
CFTC_COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

TWELVEDATA_BASE = "https://api.twelvedata.com/time_series"
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TWELVEDATA_XAU_SYMBOL = "XAU/USD"
TWELVEDATA_INTERVAL_MAP = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1day",
}
TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400,
}

SYMBOL_ALIASES = {
    "XAUUSD": "PAXGUSDT",
    "XAU": "PAXGUSDT",
    "GOLD": "PAXGUSDT",
    "PAXG": "PAXGUSDT",
}
SPOT_CHECK = {
    "PAXGUSDT": (GOLD_SPOT_URL, "XAUUSD spot"),
}
TIMEFRAMES = {
    "M1": ("1m", "M1"), "M5": ("5m", "M5"), "M15": ("15m", "M15"),
    "M30": ("30m", "M30"), "H1": ("1h", "H1"), "H4": ("4h", "H4"), "D1": ("1d", "D1"),
    "1M": ("1m", "M1"), "5M": ("5m", "M5"), "15M": ("15m", "M15"), "30M": ("30m", "M30"),
    "1H": ("1h", "H1"), "4H": ("4h", "H4"), "1D": ("1d", "D1"),
}

# Sesi overlap London-NY (UTC). Dipakai untuk anchor VWAP XAU.
LONDON_NY_OVERLAP_START_UTC = 12
LONDON_NY_OVERLAP_END_UTC = 16

NEUTRAL_THRESHOLD = 25          # zona skor -25..+25 dianggap NEUTRAL
LIQUIDITY_TOL_PCT = 0.0015      # toleransi 0.15% untuk deteksi equal high/low
FUNDING_EXTREME = 0.0005        # 0.05% dianggap funding "panas"

# --- SOP Order Block (lihat dokumen SOP proyek) -----------------------------
OB_IMPULSE_BODY_MULT = 1.5      # candle "impulsif" = body >= 1.5x rata-rata body 20 candle
                                 # (2026-09-22: sempat diturunkan ke 1.2x, tapi itu dikalibrasi
                                 # SAAT bos_idx masih bug (selalu = candle terakhir, bukan candle
                                 # breakout asli) -- data kalibrasinya tidak valid. Dikembalikan ke
                                 # 1.5x (nilai desain awal) supaya re-test dari kondisi bersih
                                 # setelah bos_idx diperbaiki. Kalibrasi ulang kalau perlu, TAPI
                                 # sampling ulang dulu -- jangan asumsikan hasil sampling lama.
OB_BUFFER_PCT = 0.0005          # 0.05% dari harga entry, buffer DI LUAR edge OB
                                 # (starting parameter -- lihat SOP: rumus tetap "edge OB +
                                 # buffer", cuma angka ini yang boleh disetel ulang nanti)

# --- Fitur baru (2026-09-22): VP refine / liquidity void / trapped traders --
# Semua pakai data yang SUDAH ditarik (klines), tidak ada API call baru.
# Semua parameter di bawah SUDAH DIKALIBRASI 2026-09-22 (lihat catatan per
# parameter). Tetap berlaku SOP OB_IMPULSE_BODY_MULT: kalau nanti ada bukti
# baru (misal validasi data H4 riil) yang bertentangan, re-kalibrasi dari
# sampling bersih -- jangan asumsikan benar selamanya dari sampling ini.
VP_HVN_RATIO = 2.0                # REVISI 2026-09-22 (v2). Riwayat: 4.0 (kalibrasi sintetis
                                  # awal) dead-on-arrival di data riil (HVN=0 selalu). 2.2
                                  # (revisi v1, tervalidasi cuma di 1 simbol/1 periode)
                                  # TERBUKTI TIDAK ROBUST begitu diuji lebih luas -- gagal
                                  # (HVN zones=0) di 2/42 window pada validasi independen:
                                  # 6 simbol/periode (PAXGUSDT/BTCUSDT/ETHUSDT x recent/older,
                                  # H4, ~1000 candle @Apr-Sep2026 & @Apr-Sep2025) x 7 sub-window
                                  # (full, last500/250/150/100, first250, middle250) = 42
                                  # kombinasi, dijalankan LANGSUNG lewat volume_profile() asli
                                  # di file ini (bukan reimplementasi terpisah). 2.0 = 0/42 gagal
                                  # -> dipilih. CATATAN JUJUR: tetap baru diuji di 3 simbol,
                                  # rentang waktu Apr2025-Sep2026 -- re-cek kalau ada aset/periode
                                  # lain yang perilakunya beda.
VP_LVN_RATIO = 0.20              # REVISI 2026-09-22 (v2). Nilai lama 0.05 (diklaim "fp rate
                                  # nyaris 0" dari 1 simbol) ternyata gagal (LVN zones=0) di
                                  # 6/42 window pada validasi independen di atas (~14%, BUKAN
                                  # "hampir selalu mati"/92% seperti sempat diklaim di sesi
                                  # chat sebelumnya -- klaim itu TIDAK bisa direproduksi saat
                                  # saya jalankan ulang volume_profile() asli terhadap data
                                  # yang sama; kemungkinan salah hitung atau salah kutip di sesi
                                  # itu, bukan bug di kode). 0.10 gagal 2/42, 0.15 gagal 1/42,
                                  # 0.20 gagal 0/42 -> dipilih karena paling bersih, meski
                                  # trade-off-nya zona LVN jadi lebih sedikit/lebih ketat
                                  # definisinya. Re-cek kalau fitur liquidity void di live
                                  # terasa terlalu jarang muncul.
TRAPPED_REVERSAL_LOOKAHEAD = 5  # dikalibrasi 2026-09-22: 3->5, konsisten top-3 di 3 sweep
                                 # (repeat 20/40/80), turunkan worst-case FP ~0.7->0.35-0.4
                                 # dengan recall rata2 tetap ~0.6-0.7 (lihat catatan struktural
                                 # di bawah soal batas noise H4). RE-VALIDASI INDEPENDEN
                                 # 2026-09-22 (v2): dijalankan langsung via find_swings() +
                                 # find_liquidity_pools() + mark_swept_pools() +
                                 # detect_trapped_traders() asli di 6 dataset H4 riil
                                 # (PAXGUSDT/BTCUSDT/ETHUSDT x recent/older) -> conversion
                                 # per dataset 33-76%, gabungan 47/91=51.6% (mirip angka yang
                                 # diklaim sesi sebelumnya, ~60% -- sedikit lebih rendah tapi
                                 # dalam rentang wajar, TIDAK ada tanda fitur mati). n=91 swept
                                 # events masih kecil untuk klaim statistik kuat; treat sebagai
                                 # indikatif, bukan final.
TRAPPED_REVERSAL_MIN_PCT = 0.002  # dikalibrasi 2026-09-22: 0.001->0.002 (lihat catatan di atas)

# Cache in-memory sederhana untuk fetch TwelveData (hindari boros quota plan Basic:
# 8 request/menit, 800/hari). TTL = durasi 1 candle sesuai timeframe yang diminta.
_TD_CACHE = {}


# =============================================================================
# 1. ALIAS / TIMEFRAME RESOLVER
# =============================================================================
def resolve_symbol(symbol: str):
    key = symbol.upper().strip()
    if key in SYMBOL_ALIASES:
        return key, SYMBOL_ALIASES[key], True
    return key, key, False


def parse_timeframe(tf: str):
    key = tf.upper().strip()
    if key in TIMEFRAMES:
        return TIMEFRAMES[key]
    raise ValueError(
        f"Timeframe '{tf}' tidak dikenal. Pilihan: M1, M5, M15, M30, H1, H4, D1 "
        f"(atau 1m, 5m, 15m, 30m, 1h, 4h, 1d)"
    )


# =============================================================================
# 2. DATA LAYER
# =============================================================================
def fetch_klines(symbol: str, interval: str = "4h", limit: int = 300) -> pd.DataFrame:
    """Ambil OHLCV dari Binance spot public API (tanpa API key)."""
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_SPOT_KLINES, params=params, timeout=15)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume", "tbb"]].copy()
    for c in ["open", "high", "low", "close", "volume", "tbb"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.iloc[:-1].reset_index(drop=True)  # buang candle berjalan
    return df


def fetch_live_price(symbol: str):
    """Harga real-time (bukan close candle terakhir yang sudah closed).
    Dipakai sebagai 'harga sekarang' / basis Entry, terpisah dari klines yang
    dipakai untuk struktur (VWAP/POC/pool/swing) -- supaya Entry tidak basi
    sampai hampir 1 candle penuh saat candle timeframe besar (H4/D1) belum
    tutup. Coba data-api.binance.vision dulu (konsisten dgn fetch_klines),
    fallback ke api.binance.com kalau gagal -- lalu log alasannya ke stderr
    supaya kelihatan di Railway logs kalau dua-duanya gagal."""
    endpoints = [
        "https://data-api.binance.vision/api/v3/ticker/price",
        "https://api.binance.com/api/v3/ticker/price",
    ]
    last_err = None
    for url in endpoints:
        try:
            r = requests.get(
                url, params={"symbol": symbol.upper()}, timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ai-trading-analyzer/1.0)"},
            )
            r.raise_for_status()
            data = r.json()
            return float(data["price"])
        except Exception as e:
            last_err = f"{url} -> {type(e).__name__}: {e}"
            continue
    print(f"[fetch_live_price] gagal untuk {symbol}: {last_err}", file=sys.stderr)
    return None


def fetch_klines_twelvedata(td_symbol: str, tf_label: str, outputsize: int = 300) -> pd.DataFrame:
    """
    Ambil candle ASLI (XAUUSD, bukan proxy) dari TwelveData. Dipakai KHUSUS
    untuk layer STRUKTUR (swing/BOS/FVG/liquidity pool/Order Block) sesuai
    SOP hybrid proyek ini. Volume/VWAP/reaksi-candle TETAP dari PAXGUSDT
    (lihat analyze_xau) karena volume forex/XAU di TwelveData tidak reliable
    -- kolom volume di sini sengaja diisi 0, JANGAN dipakai untuk apa pun.

    Caching in-memory per (symbol, interval) dengan TTL = durasi 1 candle,
    supaya hemat quota (plan Basic = 8 request/menit, 800/hari).
    Melempar exception kalau API key belum ada / request gagal / limit habis
    -- pemanggil (get_structure_df) yang bertanggung jawab fallback.
    """
    td_interval = TWELVEDATA_INTERVAL_MAP.get(tf_label)
    if not td_interval:
        raise ValueError(f"Timeframe '{tf_label}' tidak dipetakan ke interval TwelveData")
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY belum di-set (env var kosong)")

    cache_key = (td_symbol, td_interval)
    ttl = TIMEFRAME_SECONDS.get(tf_label, 3600)
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _TD_CACHE.get(cache_key)
    if cached and (now_ts - cached["ts"]) < ttl:
        return cached["df"].copy()

    params = {
        "symbol": td_symbol, "interval": td_interval, "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY, "order": "ASC",
    }
    r = requests.get(TWELVEDATA_BASE, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict) or "values" not in data:
        raise RuntimeError(f"TwelveData response tidak valid: {data}")

    df = pd.DataFrame(data["values"]).rename(columns={"datetime": "open_time"})
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["volume"] = 0.0   # sengaja kosong -- lihat catatan di docstring
    df["tbb"] = 0.0
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df[["open_time", "open", "high", "low", "close", "volume", "tbb"]]

    _TD_CACHE[cache_key] = {"df": df.copy(), "ts": now_ts}
    return df


def get_structure_df(asset_class: str, tf_label: str, fallback_df: pd.DataFrame):
    """
    Tentukan DataFrame mana yang dipakai untuk layer struktur (swing/BOS/FVG/
    liquidity pool/Order Block), sesuai SOP hybrid:
      - crypto  -> selalu df Binance sendiri (sudah data asli, tidak perlu TwelveData)
      - xau     -> coba TwelveData (XAUUSD asli) dulu; kalau API key belum ada,
                   request gagal, atau data terlalu sedikit -> fallback ke
                   fallback_df (candle PAXGUSDT yang sudah ada), source
                   ditandai eksplisit supaya laporan tidak menyamarkan ini
                   sebagai struktur XAUUSD asli.
    Return: (df_structure, structure_source)
    """
    if asset_class != "xau":
        return fallback_df, "native"
    if not TWELVEDATA_API_KEY:
        return fallback_df, "paxg_fallback_no_api_key"
    try:
        df_td = fetch_klines_twelvedata(TWELVEDATA_XAU_SYMBOL, tf_label, outputsize=300)
        if len(df_td) < 30:
            return fallback_df, "paxg_fallback_insufficient_data"
        return df_td, "twelvedata"
    except Exception as e:
        print(f"[get_structure_df] TwelveData gagal, fallback ke PAXGUSDT: {e}", file=sys.stderr)
        return fallback_df, "paxg_fallback_error"


def fetch_spot_crosscheck(binance_symbol: str):
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


def symbol_has_futures(symbol: str) -> bool:
    """Cek apakah symbol punya kontrak USDT-M futures di Binance (buat funding/OI)."""
    try:
        r = requests.get(FAPI_PREMIUM_INDEX, params={"symbol": symbol.upper()}, timeout=10)
        if r.status_code != 200:
            return False
        d = r.json()
        return isinstance(d, dict) and "lastFundingRate" in d
    except Exception:
        return False


def fetch_funding_rate(symbol: str):
    """Funding rate terkini. REAL DATA, gratis, tanpa API key (Binance Futures)."""
    try:
        r = requests.get(FAPI_PREMIUM_INDEX, params={"symbol": symbol.upper()}, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "rate": float(d.get("lastFundingRate", 0.0)),
            "mark_price": float(d.get("markPrice", 0.0)),
            "next_funding_time": d.get("nextFundingTime"),
        }
    except Exception:
        return None


def fetch_open_interest(symbol: str, period="1h", limit=24):
    """Open Interest terkini + histori singkat. REAL DATA, gratis, tanpa API key."""
    out = {"current": None, "series": [], "trend": None}
    try:
        r = requests.get(FAPI_OPEN_INTEREST, params={"symbol": symbol.upper()}, timeout=10)
        r.raise_for_status()
        out["current"] = float(r.json().get("openInterest", 0.0))
    except Exception:
        pass
    try:
        r2 = requests.get(FAPI_OI_HIST,
                           params={"symbol": symbol.upper(), "period": period, "limit": limit},
                           timeout=10)
        r2.raise_for_status()
        hist = r2.json()
        series = [float(x["sumOpenInterest"]) for x in hist] if hist else []
        out["series"] = series
        if len(series) >= 2:
            out["trend"] = series[-1] - series[0]
    except Exception:
        pass
    return out


def compute_usdx(rates: dict) -> float:
    """Formula resmi ICE USDX dari kurs USD->{EUR,JPY,GBP,CAD,SEK,CHF}."""
    eur_usd = 1.0 / rates["EUR"]
    gbp_usd = 1.0 / rates["GBP"]
    jpy, cad, sek, chf = rates["JPY"], rates["CAD"], rates["SEK"], rates["CHF"]
    return (50.14348112
            * (eur_usd ** -0.576)
            * (jpy ** 0.136)
            * (gbp_usd ** -0.119)
            * (cad ** 0.091)
            * (sek ** 0.042)
            * (chf ** 0.036))


def fetch_dxy_proxy(trend_days=5):
    """
    Proxy DXY dari kurs referensi ECB (frankfurter.app, gratis tanpa key).
    Update HARIAN, bukan real-time intraday -> hanya untuk bias makro kasar.
    """
    try:
        r = requests.get(FX_LATEST_URL,
                          params={"from": "USD", "to": "EUR,JPY,GBP,CAD,SEK,CHF"}, timeout=10)
        r.raise_for_status()
        rates_now = r.json()["rates"]
        dxy_now = compute_usdx(rates_now)

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=trend_days + 3)  # buffer weekend
        r2 = requests.get(FX_RANGE_URL.format(start=start.isoformat(), end=end.isoformat()),
                           params={"from": "USD", "to": "EUR,JPY,GBP,CAD,SEK,CHF"}, timeout=10)
        r2.raise_for_status()
        series = r2.json().get("rates", {})
        dxy_hist = {d: compute_usdx(v) for d, v in series.items()}
        dates_sorted = sorted(dxy_hist)
        change_pct = None
        if len(dates_sorted) >= 2:
            first_v = dxy_hist[dates_sorted[0]]
            change_pct = (dxy_now - first_v) / first_v * 100.0
        return {"dxy": dxy_now, "change_pct": change_pct,
                "note": "proxy harian via ECB reference rate (frankfurter.app), bukan intraday real-time"}
    except Exception:
        return None


def fetch_cot_gold():
    """
    COT report Gold (CFTC, gratis tanpa key) -> INFORMASIONAL SAJA, mingguan/delayed.
    TIDAK dipakai untuk skor. Best-effort: kalau struktur endpoint berubah, gagal senyap.
    """
    try:
        params = {
            "$where": "market_and_exchange_names like 'GOLD - COMMODITY EXCHANGE%'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": 1,
        }
        r = requests.get(CFTC_COT_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        row = data[0]
        return {
            "date": row.get("report_date_as_yyyy_mm_dd"),
            "noncomm_long": row.get("noncomm_positions_long_all"),
            "noncomm_short": row.get("noncomm_positions_short_all"),
            "comm_long": row.get("comm_positions_long_all"),
            "comm_short": row.get("comm_positions_short_all"),
        }
    except Exception:
        return None


# =============================================================================
# 3. STRUKTUR & LIKUIDITAS LAYER (pengganti EMA/RSI/MACD/BB/ATR/Ichimoku)
# =============================================================================
def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Deteksi swing high/low pakai metode fraktal (bukan indikator lagging)."""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(left, n - right):
        wh = highs[i - left:i + right + 1]
        if highs[i] == wh.max() and (wh == highs[i]).sum() == 1:
            swing_highs.append((i, df["open_time"].iloc[i], float(highs[i])))
        wl = lows[i - left:i + right + 1]
        if lows[i] == wl.min() and (wl == lows[i]).sum() == 1:
            swing_lows.append((i, df["open_time"].iloc[i], float(lows[i])))
    return swing_highs, swing_lows


def classify_structure(df: pd.DataFrame, swing_highs, swing_lows, n_last: int = 3):
    """
    Klasifikasi struktur HTF: bullish (HH+HL), bearish (LH+LL), atau ranging.
    Plus deteksi Break of Structure (BOS) sederhana: close terakhir menembus
    swing high/low signifikan terakhir.
    """
    result = {"bias": "ranging", "detail": "", "bos": None, "bos_idx": None}
    if len(swing_highs) >= 2:
        last_h = [p for _, _, p in swing_highs[-n_last:]]
        hh = all(last_h[i] < last_h[i + 1] for i in range(len(last_h) - 1))
        lh = all(last_h[i] > last_h[i + 1] for i in range(len(last_h) - 1))
    else:
        hh = lh = False
    if len(swing_lows) >= 2:
        last_l = [p for _, _, p in swing_lows[-n_last:]]
        hl = all(last_l[i] < last_l[i + 1] for i in range(len(last_l) - 1))
        ll = all(last_l[i] > last_l[i + 1] for i in range(len(last_l) - 1))
    else:
        hl = ll = False

    if hh and hl:
        result["bias"] = "bullish"
        result["detail"] = "Higher-High & Higher-Low (struktur uptrend)"
    elif lh and ll:
        result["bias"] = "bearish"
        result["detail"] = "Lower-High & Lower-Low (struktur downtrend)"
    else:
        result["detail"] = "Struktur campuran / konsolidasi (belum ada HH-HL atau LH-LL bersih)"

    last_close = df["close"].iloc[-1]
    last_idx = len(df) - 1

    def _first_break_idx(swing_idx: int, level: float, is_bullish: bool):
        """Cari candle PERTAMA (bukan candle terakhir df) yang close-nya benar-benar
        menembus level, dimulai tepat setelah swing itu terbentuk. Ini PENTING karena
        find_order_blocks() mundur dari bos_idx untuk cari candle impulsif -- kalau
        bos_idx selalu dipaksa ke candle terakhir (live 'now'), OB yang terdeteksi jadi
        tidak konsisten dan tidak benar-benar terkait dengan breakout aslinya."""
        for i in range(swing_idx + 1, last_idx + 1):
            c = df["close"].iloc[i]
            if (is_bullish and c > level) or (not is_bullish and c < level):
                return i
        return None

    if swing_highs:
        sh_idx, _, last_swing_high = swing_highs[-1]
        if last_close > last_swing_high:
            bi = _first_break_idx(sh_idx, last_swing_high, True)
            if bi is not None:
                result["bos"] = ("bullish", last_swing_high)
                result["bos_idx"] = bi
    if swing_lows:
        sl_idx, _, last_swing_low = swing_lows[-1]
        if last_close < last_swing_low:
            bi = _first_break_idx(sl_idx, last_swing_low, False)
            if bi is not None:
                # kalau dua-duanya break (jarang), yang index breakout-nya paling
                # BARU yang menang (bukan sekadar swing index-nya)
                if result["bos"] is None or bi >= result["bos_idx"]:
                    result["bos"] = ("bearish", last_swing_low)
                    result["bos_idx"] = bi
    return result


def find_liquidity_pools(swing_highs, swing_lows, tol_pct: float = LIQUIDITY_TOL_PCT):
    """
    ERL (External Range Liquidity): cluster equal-high / equal-low -> lokasi
    stop-loss retail menumpuk -> jadi 'magnet' harga (liquidity pool).
    """
    def cluster(points, side):
        pools = []
        used = [False] * len(points)
        for i in range(len(points)):
            if used[i]:
                continue
            base_price = points[i][2]
            touches = [points[i]]
            used[i] = True
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                if abs(points[j][2] - base_price) / base_price <= tol_pct:
                    touches.append(points[j])
                    used[j] = True
            if len(touches) >= 2:
                avg_price = float(np.mean([t[2] for t in touches]))
                pools.append({
                    "price": avg_price, "side": side, "touches": len(touches),
                    "last_idx": max(t[0] for t in touches),
                })
        return pools

    buyside = cluster(swing_highs, "buyside")   # liquidity DI ATAS harga (equal highs)
    sellside = cluster(swing_lows, "sellside")  # liquidity DI BAWAH harga (equal lows)
    return buyside + sellside


def mark_swept_pools(df: pd.DataFrame, pools: list):
    """Tandai pool yang sudah 'disapu' (wick tembus lalu close kembali) -> liquidity grab."""
    for p in pools:
        seg = df.iloc[p["last_idx"] + 1:]
        if seg.empty:
            p["swept"] = False
            continue
        if p["side"] == "buyside":
            wicked = (seg["high"] > p["price"]).any()
            closed_back = (seg["close"] < p["price"]).iloc[-1] if wicked else False
        else:
            wicked = (seg["low"] < p["price"]).any()
            closed_back = (seg["close"] > p["price"]).iloc[-1] if wicked else False
        p["swept"] = bool(wicked and closed_back)
    return pools


def find_fair_value_gaps(df: pd.DataFrame, max_lookback: int = 60):
    """
    IRL (Internal Range Liquidity) via Fair Value Gap / imbalance 3-candle.
    Zona ini bertindak sebagai magnet 'inefisiensi harga' yang cenderung ditarik.
    """
    fvgs = []
    start = max(2, len(df) - max_lookback)
    for i in range(start, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvgs.append({"type": "bullish", "zone_low": float(df["high"].iloc[i - 2]),
                         "zone_high": float(df["low"].iloc[i]), "idx": i})
        if df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvgs.append({"type": "bearish", "zone_low": float(df["high"].iloc[i]),
                         "zone_high": float(df["low"].iloc[i - 2]), "idx": i})
    # hanya simpan yang belum "terisi penuh" oleh candle sesudahnya
    unfilled = []
    for g in fvgs:
        seg = df.iloc[g["idx"] + 1:]
        if g["type"] == "bullish":
            filled = (seg["low"] <= g["zone_low"]).any() if not seg.empty else False
        else:
            filled = (seg["high"] >= g["zone_high"]).any() if not seg.empty else False
        g["filled"] = bool(filled)
        if not filled:
            unfilled.append(g)
    return unfilled


def find_order_blocks(df: pd.DataFrame, bos, bos_idx, body_mult: float = OB_IMPULSE_BODY_MULT,
                       max_lookback: int = 40):
    """
    Order Block sesuai SOP proyek ini (BUKAN ICT murni): candle BERLAWANAN
    warna TERAKHIR sebelum candle IMPULSIF yang memicu BOS. FVG saja tanpa
    BOS TIDAK dianggap OB valid -- makanya fungsi ini butuh `bos`/`bos_idx`
    dari classify_structure(), bukan cuma FVG independen.

    "Candle impulsif" = body candle >= body_mult x rata-rata body 20 candle
    sebelumnya, DAN searah BOS (BOS bullish -> candle impulsif naik, dst).

    Return list (biasanya 0 atau 1 item) supaya pola konsisten dengan
    find_fair_value_gaps/find_liquidity_pools.
    """
    if bos is None or bos_idx is None:
        return []
    direction, _break_level = bos

    bodies = (df["close"] - df["open"]).abs()
    hist_start = max(0, bos_idx - 20)
    avg_body = bodies.iloc[hist_start:bos_idx].mean()
    if not avg_body or avg_body <= 0 or np.isnan(avg_body):
        return []

    lookback_start = max(0, bos_idx - max_lookback)

    # 1) cari candle impulsif: mundur dari bos_idx, body >= body_mult x avg_body,
    #    warnanya searah BOS
    impulse_idx = None
    for i in range(bos_idx, lookback_start - 1, -1):
        is_up = df["close"].iloc[i] > df["open"].iloc[i]
        same_dir = (direction == "bullish" and is_up) or (direction == "bearish" and not is_up)
        if same_dir and bodies.iloc[i] >= body_mult * avg_body:
            impulse_idx = i
            break
    if impulse_idx is None:
        return []

    # 2) OB = candle berlawanan warna TERAKHIR sebelum candle impulsif tsb
    ob_idx = None
    for j in range(impulse_idx - 1, lookback_start - 1, -1):
        is_up_j = df["close"].iloc[j] > df["open"].iloc[j]
        if direction == "bullish" and not is_up_j:
            ob_idx = j
            break
        if direction == "bearish" and is_up_j:
            ob_idx = j
            break
    if ob_idx is None:
        return []

    return [{
        "type": "bullish" if direction == "bullish" else "bearish",
        "high": float(df["high"].iloc[ob_idx]),
        "low": float(df["low"].iloc[ob_idx]),
        "idx": ob_idx,
        "impulse_idx": impulse_idx,
        "open_time": df["open_time"].iloc[ob_idx],
    }]


def mark_mitigated_ob(df: pd.DataFrame, obs: list):
    """
    OB dianggap 'mitigated' kalau harga SUDAH balik masuk ke dalam zonanya
    setelah OB terbentuk. OB yang sudah mitigated dianggap tidak lagi valid
    dipakai sebagai entry baru (sudah "dipakai" sekali) -- pola sama seperti
    mark_swept_pools().

    PENTING: pengecekan dimulai SETELAH candle IMPULSIF (bukan setelah OB itu
    sendiri) -- candle impulsif biasanya open persis di level close OB, jadi
    kalau start-nya dari ob['idx']+1, low candle impulsif nyaris selalu
    "menyentuh" tepi OB dan salah dianggap mitigated padahal itu cuma titik
    awal impulsnya sendiri, bukan retest asli.
    """
    for ob in obs:
        start_idx = ob.get("impulse_idx", ob["idx"]) + 1
        seg = df.iloc[start_idx:]
        if seg.empty:
            ob["mitigated"] = False
            continue
        if ob["type"] == "bullish":
            touched = (seg["low"] <= ob["high"]).any()
        else:
            touched = (seg["high"] >= ob["low"]).any()
        ob["mitigated"] = bool(touched)
    return obs


def classify_tier(risk: float, tp1_dist: float, tp2_dist: float) -> str:
    """
    Tiering kelayakan setup sesuai SOP:
      A = risk <= jarak TP1  -> layak penuh
      B = risk <= jarak TP2  -> layak marjinal (TP1 belum menutup risiko)
      C = risk > jarak TP2   -> berisiko tinggi
    Semua tier tetap ditampilkan di report, tidak ada yang disembunyikan.
    """
    if risk <= 0:
        return "C"
    if risk <= tp1_dist:
        return "A"
    elif risk <= tp2_dist:
        return "B"
    return "C"


def volume_profile(df: pd.DataFrame, bins: int = 24, value_area_pct: float = 0.70):
    """
    Volume Profile dari OHLCV (approksimasi: volume tiap candle dibebankan ke
    bin harga typical price (H+L+C)/3 -- standar untuk data candle, bukan tick).
    Menghasilkan POC (Point of Control), VAH, VAL.
    """
    price_min, price_max = df["low"].min(), df["high"].max()
    if price_max <= price_min:
        price_max = price_min * 1.0001
    edges = np.linspace(price_min, price_max, bins + 1)
    vol_bins = np.zeros(bins)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    bin_idx = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)
    for idx, vol in zip(bin_idx, df["volume"]):
        vol_bins[idx] += vol

    poc_bin = int(np.argmax(vol_bins))
    poc_price = float((edges[poc_bin] + edges[poc_bin + 1]) / 2)

    total_vol = vol_bins.sum()
    target = total_vol * value_area_pct
    included = {poc_bin}
    acc = vol_bins[poc_bin]
    lo, hi = poc_bin, poc_bin
    while acc < target and (lo > 0 or hi < bins - 1):
        left_vol = vol_bins[lo - 1] if lo > 0 else -1
        right_vol = vol_bins[hi + 1] if hi < bins - 1 else -1
        if right_vol >= left_vol and hi < bins - 1:
            hi += 1
            acc += vol_bins[hi]
            included.add(hi)
        elif lo > 0:
            lo -= 1
            acc += vol_bins[lo]
            included.add(lo)
        else:
            break
    vah = float(edges[hi + 1])
    val = float(edges[lo])

    # --- Refinement: High/Low Volume Node (HVN/LVN) --------------------------
    # HVN = bin volume jauh di atas rata-rata -> zona "acceptance" (harga
    #       cenderung berlama-lama/rotate di situ).
    # LVN = bin volume jauh di bawah rata-rata -> zona harga dilewati cepat,
    #       jadi kandidat "liquidity void" (lihat find_liquidity_voids()).
    nonzero = vol_bins[vol_bins > 0]
    avg_bin_vol = float(nonzero.mean()) if len(nonzero) else 0.0
    hvn_zones, lvn_zones = [], []
    if avg_bin_vol > 0:
        for i in range(bins):
            v = float(vol_bins[i])
            zone = {"price_low": float(edges[i]), "price_high": float(edges[i + 1]), "volume": v}
            if v >= VP_HVN_RATIO * avg_bin_vol:
                hvn_zones.append(zone)
            elif v <= VP_LVN_RATIO * avg_bin_vol:
                lvn_zones.append(zone)
    hvn_zones = _merge_adjacent_zones(hvn_zones)
    lvn_zones = _merge_adjacent_zones(lvn_zones)

    return {"poc": poc_price, "vah": vah, "val": val, "edges": edges, "vol_bins": vol_bins,
            "hvn": hvn_zones, "lvn": lvn_zones}


def _merge_adjacent_zones(zones: list):
    """Gabungkan bin-bin harga yang bersebelahan (edge nyambung) jadi satu zona lebih lebar."""
    if not zones:
        return []
    merged = [dict(zones[0])]
    for z in zones[1:]:
        last = merged[-1]
        if abs(z["price_low"] - last["price_high"]) < 1e-12:
            last["price_high"] = z["price_high"]
            last["volume"] += z["volume"]
        else:
            merged.append(dict(z))
    return merged


def find_liquidity_voids(price: float, vp: dict, max_zones: int = 3):
    """
    Liquidity void = zona LVN (Low Volume Node) dari volume profile di atas --
    area harga yang historisnya dilewati cepat dengan sedikit volume, jadi
    "kosong" dan cenderung ditarik/diisi cepat kalau harga lewat situ lagi.

    Basisnya DISTRIBUSI VOLUME per level harga (vp["lvn"]), BUKAN gap 3-candle
    OHLC seperti find_fair_value_gaps() -- jadi ini fitur genuine baru, bukan
    relabeling FVG, meski konsepnya serupa ("magnet" harga).
    """
    voids = []
    for z in vp.get("lvn", []):
        mid = (z["price_low"] + z["price_high"]) / 2
        voids.append({
            "price_low": z["price_low"], "price_high": z["price_high"],
            "mid": mid, "volume": z["volume"],
            "side": "above" if mid > price else "below",
            "width_pct": ((z["price_high"] - z["price_low"]) / price) if price else 0.0,
        })
    voids.sort(key=lambda v: abs(v["mid"] - price))
    return voids[:max_zones]


def detect_trapped_traders(df: pd.DataFrame, pools: list,
                            lookahead: int = TRAPPED_REVERSAL_LOOKAHEAD,
                            min_reversal_pct: float = TRAPPED_REVERSAL_MIN_PCT):
    """
    Trapped buyers/sellers -- dibangun DI ATAS mark_swept_pools() yang sudah
    ada (bukan data baru, bukan order book): pool yang disapu (swept) lalu
    harga GAGAL lanjut & reverse balik arah dalam `lookahead` candle = jejak
    trader yang entry di breakout palsu dan sekarang "terjebak".

    - Sweep buyside (equal high disapu) lalu reverse turun -> TRAPPED BUYERS.
    - Sweep sellside (equal low disapu) lalu reverse naik  -> TRAPPED SELLERS.

    df harus konsisten dengan `pools` yang dipakai (struct_df untuk XAU,
    df biasa untuk crypto) -- sama seperti mark_swept_pools().
    """
    events = []
    for p in pools:
        if not p.get("swept"):
            continue
        seg = df.iloc[p["last_idx"] + 1:]
        if seg.empty:
            continue
        sweep_mask = (seg["high"] > p["price"]) if p["side"] == "buyside" else (seg["low"] < p["price"])
        if not sweep_mask.any():
            continue
        sweep_pos = sweep_mask.idxmax()
        sweep_iloc = df.index.get_loc(sweep_pos)
        after = df.iloc[sweep_iloc + 1: sweep_iloc + 1 + lookahead]
        if after.empty:
            continue
        last_close = float(after["close"].iloc[-1])
        kind = None
        if p["side"] == "buyside" and last_close < p["price"] * (1 - min_reversal_pct):
            kind = "trapped_buyers"
        elif p["side"] == "sellside" and last_close > p["price"] * (1 + min_reversal_pct):
            kind = "trapped_sellers"
        if kind:
            events.append({"kind": kind, "pool_price": p["price"], "pool_side": p["side"],
                            "touches": p["touches"], "confirm_close": last_close})
    return events


def anchored_vwap(df: pd.DataFrame, anchor_ts=None):
    """VWAP dari titik anchor tertentu (mis. awal sesi). Kalau anchor_ts None -> seluruh df."""
    seg = df if anchor_ts is None else df[df["open_time"] >= anchor_ts]
    if seg.empty:
        seg = df
    typical = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    cum_pv = (typical * seg["volume"]).cumsum()
    cum_v = seg["volume"].cumsum()
    vwap_series = cum_pv / cum_v.replace(0, np.nan)
    return float(vwap_series.iloc[-1]), vwap_series


def session_window_today(df: pd.DataFrame, start_hour_utc: int, end_hour_utc: int):
    """Ambil bar-bar yang jatuh di jendela sesi (UTC) hari terakhir yang tersedia."""
    last_ts = df["open_time"].iloc[-1]
    day = last_ts.date()
    start_ts = pd.Timestamp(datetime(day.year, day.month, day.day, start_hour_utc, tzinfo=timezone.utc))
    end_ts = pd.Timestamp(datetime(day.year, day.month, day.day, end_hour_utc, tzinfo=timezone.utc))
    seg = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)]
    if seg.empty:  # kalau sesi hari ini belum lewat / data kurang granular, mundur 1 hari
        start_ts -= timedelta(days=1)
        end_ts -= timedelta(days=1)
        seg = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)]
    return seg, start_ts


# =============================================================================
# 4. ORDER FLOW (tetap dipakai — ini BUKAN indikator lagging klasik,
#    melainkan derivasi langsung dari taker-buy-volume di klines Binance)
# =============================================================================
def orderflow_delta(df: pd.DataFrame) -> pd.Series:
    taker_buy = df["tbb"]
    taker_sell = df["volume"] - df["tbb"]
    return taker_buy - taker_sell


def cvd(delta: pd.Series) -> pd.Series:
    return delta.cumsum()


def liquidation_magnet_proxy(price, pools, funding, oi):
    """
    PROXY 'liquidation heatmap' (BUKAN data notional likuidasi riil).
    Menilai pool likuiditas mana yang paling mungkin jadi target 'sapuan'
    berikutnya, diperkuat oleh funding ekstrem & tren OI.
    """
    if not pools:
        return None
    unswept = [p for p in pools if not p.get("swept", False)]
    if not unswept:
        return None
    # pool terdekat di tiap sisi
    above = [p for p in unswept if p["price"] > price]
    below = [p for p in unswept if p["price"] < price]
    nearest_above = min(above, key=lambda p: p["price"] - price) if above else None
    nearest_below = min(below, key=lambda p: price - p["price"]) if below else None

    funding_rate = funding["rate"] if funding else None
    oi_trend = oi["trend"] if oi else None

    # direction = posisi pool RELATIF terhadap harga ("atas"/"bawah").
    # pool["side"] = tag ASLI pool ("buyside"/"sellside", dari swing high/low
    # pembentuknya) -> dua hal ini TIDAK BOLEH digabung jadi satu label,
    # karena pool "buyside" (dari swing high lama) bisa saja sekarang berada
    # DI BAWAH harga kalau harga sudah breakout naik melewatinya.
    direction, magnet_pool, strength_notes = None, None, []
    if funding_rate is not None and funding_rate > FUNDING_EXTREME and nearest_below:
        direction, magnet_pool = "bawah", nearest_below
        strength_notes.append(f"funding rate panas positif ({funding_rate:+.4%}) -> long crowded, "
                               f"rawan likuidasi ke bawah")
    elif funding_rate is not None and funding_rate < -FUNDING_EXTREME and nearest_above:
        direction, magnet_pool = "atas", nearest_above
        strength_notes.append(f"funding rate negatif ekstrem ({funding_rate:+.4%}) -> short crowded, "
                               f"rawan short-squeeze ke atas")
    elif nearest_below and (not nearest_above or (price - nearest_below["price"]) < (nearest_above["price"] - price)):
        direction, magnet_pool = "bawah", nearest_below
        strength_notes.append("liquidity pool terdekat berada di bawah harga")
    elif nearest_above:
        direction, magnet_pool = "atas", nearest_above
        strength_notes.append("liquidity pool terdekat berada di atas harga")

    if oi_trend is not None:
        if oi_trend > 0:
            strength_notes.append("OI naik -> posisi baru terus dibuka, tekanan ke magnet menguat")
        else:
            strength_notes.append("OI turun -> ada penutupan posisi / likuidasi berjalan")

    if magnet_pool is None:
        return None
    return {
        "direction": direction,                    # "atas" / "bawah" relatif harga saat ini
        "pool_side": magnet_pool["side"],           # tag asli pool: "buyside" / "sellside"
        "price": magnet_pool["price"],
        "touches": magnet_pool["touches"], "notes": strength_notes,
    }


# =============================================================================
# 5. SCORING ENGINE — CRYPTO
# =============================================================================
def analyze_crypto(df: pd.DataFrame, symbol: str, has_futures: bool) -> dict:
    price = float(df["close"].iloc[-1])
    swing_h, swing_l = find_swings(df, left=3, right=3)
    structure = classify_structure(df, swing_h, swing_l)
    pools = find_liquidity_pools(swing_h, swing_l)
    pools = mark_swept_pools(df, pools)
    fvgs = find_fair_value_gaps(df)
    order_blocks = find_order_blocks(df, structure.get("bos"), structure.get("bos_idx"))
    order_blocks = mark_mitigated_ob(df, order_blocks)
    vp = volume_profile(df.tail(150))
    vwap_val, _ = anchored_vwap(df.tail(96))  # ~ jendela rolling, crypto 24/7

    delta = orderflow_delta(df)
    cvd_s = cvd(delta)
    delta_now = float(delta.iloc[-1])
    lookback = min(10, len(cvd_s) - 1)
    cvd_trend = float(cvd_s.iloc[-1] - cvd_s.iloc[-1 - lookback])
    price_change_lb = float(df["close"].iloc[-1] - df["close"].iloc[-1 - lookback])

    funding = fetch_funding_rate(symbol) if has_futures else None
    oi = fetch_open_interest(symbol) if has_futures else None
    magnet = liquidation_magnet_proxy(price, pools, funding, oi)
    trapped = detect_trapped_traders(df, pools)
    voids = find_liquidity_voids(price, vp)

    score = 0.0
    reasons = []

    # 1) Struktur HTF
    if structure["bias"] == "bullish":
        score += 25
    elif structure["bias"] == "bearish":
        score -= 25
    reasons.append(f"Struktur: {structure['detail']}")
    if structure["bos"]:
        side, lvl = structure["bos"]
        score += 10 if side == "bullish" else -10
        reasons.append(f"Break of Structure {side} menembus level {lvl:.4f}")
    if order_blocks:
        ob0 = order_blocks[0]
        ob_status = "sudah dimitigasi" if ob0.get("mitigated") else "fresh, belum dimitigasi"
        reasons.append(f"Order Block {ob0['type']} @ {ob0['low']:.4f}-{ob0['high']:.4f} ({ob_status}) "
                        f"-- info struktural umum, BELUM TENTU dipakai untuk entry; "
                        f"cek baris 'Order Block dipakai' di atas (kosong = tidak dipakai/FALLBACK)")

    # 2) Liquidity pool / magnet (ERL) + FVG (IRL)
    if magnet:
        score += 15 if magnet["direction"] == "atas" else -15
        reasons.append(f"Liquidity magnet {magnet['direction']} harga @ {magnet['price']:.4f} "
                        f"(pool {magnet['pool_side']}, {magnet['touches']}x equal level) — "
                        + "; ".join(magnet["notes"]))
    unfilled_bull = [g for g in fvgs if g["type"] == "bullish"]
    unfilled_bear = [g for g in fvgs if g["type"] == "bearish"]
    if unfilled_bull and price > unfilled_bull[-1]["zone_high"]:
        score += 8
        reasons.append(f"FVG bullish belum terisi di {unfilled_bull[-1]['zone_low']:.4f}-"
                        f"{unfilled_bull[-1]['zone_high']:.4f} -> berfungsi sebagai IRL support "
                        f"di bawah harga, uptrend masih sehat")
    if unfilled_bear and price < unfilled_bear[-1]["zone_low"]:
        score -= 8
        reasons.append(f"FVG bearish belum terisi di {unfilled_bear[-1]['zone_low']:.4f}-"
                        f"{unfilled_bear[-1]['zone_high']:.4f} -> berfungsi sebagai IRL resistance "
                        f"di atas harga, downtrend masih sehat")

    # 3) VWAP
    if price > vwap_val:
        score += 15
        reasons.append(f"Harga di atas VWAP ({vwap_val:.4f})")
    else:
        score -= 15
        reasons.append(f"Harga di bawah VWAP ({vwap_val:.4f})")

    # 4) Volume Profile / POC
    if price > vp["vah"]:
        score += 15
        reasons.append(f"Harga di atas Value Area High ({vp['vah']:.4f}) -> ekstensi bullish")
    elif price < vp["val"]:
        score -= 15
        reasons.append(f"Harga di bawah Value Area Low ({vp['val']:.4f}) -> ekstensi bearish")
    elif price > vp["poc"]:
        score += 7
        reasons.append(f"Harga di atas POC ({vp['poc']:.4f}), masih dalam value area")
    else:
        score -= 7
        reasons.append(f"Harga di bawah POC ({vp['poc']:.4f}), masih dalam value area")

    # 5) Order flow (delta + CVD divergence)
    if delta_now > 0:
        score += 8
        reasons.append(f"Delta candle terakhir {delta_now:+.2f} -> taker buy dominan")
    elif delta_now < 0:
        score -= 8
        reasons.append(f"Delta candle terakhir {delta_now:+.2f} -> taker sell dominan")
    if cvd_trend > 0 and price_change_lb > 0:
        score += 10
        reasons.append("CVD naik & harga naik -> order flow konfirmasi uptrend")
    elif cvd_trend < 0 and price_change_lb < 0:
        score -= 10
        reasons.append("CVD turun & harga turun -> order flow konfirmasi downtrend")
    elif cvd_trend < 0 and price_change_lb > 0:
        score -= 15
        reasons.append("Harga naik tapi CVD turun -> divergence bearish (distribusi tersembunyi)")
    elif cvd_trend > 0 and price_change_lb < 0:
        score += 15
        reasons.append("Harga turun tapi CVD naik -> divergence bullish (akumulasi tersembunyi)")

    # 6) Funding rate
    if funding:
        rate = funding["rate"]
        if rate > FUNDING_EXTREME:
            score -= 12
            reasons.append(f"Funding rate {rate:+.4%} -> long crowded, risiko downside likuidasi")
        elif rate < -FUNDING_EXTREME:
            score += 12
            reasons.append(f"Funding rate {rate:+.4%} -> short crowded, risiko short-squeeze")
        else:
            reasons.append(f"Funding rate {rate:+.4%} -> netral, belum ekstrem")
    else:
        reasons.append("Funding rate: tidak tersedia (kontrak futures tidak ditemukan)")

    # 7) Open Interest trend
    if oi and oi["trend"] is not None:
        if oi["trend"] > 0 and price_change_lb > 0:
            score += 10
            reasons.append("OI naik + harga naik -> longs baru masuk, tren terkonfirmasi")
        elif oi["trend"] > 0 and price_change_lb < 0:
            score -= 10
            reasons.append("OI naik + harga turun -> shorts baru masuk, tren terkonfirmasi")
        elif oi["trend"] < 0:
            reasons.append("OI turun -> posisi closing/likuidasi berjalan, potensi exhaustion")
    else:
        reasons.append("Open Interest: data tidak tersedia")

    # 8) Trapped buyers/sellers (fitur baru, lihat detect_trapped_traders())
    for ev in trapped:
        if ev["kind"] == "trapped_buyers":
            score -= 10
            reasons.append(f"Trapped buyers: buyside liquidity @ {ev['pool_price']:.4f} disapu "
                            f"({ev['touches']}x equal high) lalu harga reverse turun (close "
                            f"{ev['confirm_close']:.4f}) -> potensi bahan bakar tekanan jual lanjutan")
        else:
            score += 10
            reasons.append(f"Trapped sellers: sellside liquidity @ {ev['pool_price']:.4f} disapu "
                            f"({ev['touches']}x equal low) lalu harga reverse naik (close "
                            f"{ev['confirm_close']:.4f}) -> potensi bahan bakar tekanan beli lanjutan")

    # 9) Liquidity void (LVN dari volume profile) -- INFORMASIONAL, TIDAK
    #    menambah skor (biar tidak dobel-hitung dengan POC/VAH/VAL di atas).
    #    Berguna sebagai referensi target/TP: harga cenderung "mengisi" void
    #    dengan cepat kalau ditarik ke situ.
    for v in voids[:2]:
        reasons.append(f"Liquidity void (LVN) {v['side']} harga di {v['price_low']:.4f}-"
                        f"{v['price_high']:.4f} (volume tipis di histori) -> kandidat target "
                        f"kalau harga bergerak ke arah situ, cenderung dilewati cepat")

    # 10) AMT (Auction Market Theory) -- INTERPRETASI di atas Volume Profile
    #     yang sudah dihitung (poin 4), BUKAN faktor skor terpisah, supaya
    #     tidak menghitung ulang sinyal VAH/VAL/POC dengan nama berbeda.
    reasons.append(f"[AMT] POC {vp['poc']:.4f} = fair value price; area value "
                    f"{vp['val']:.4f}-{vp['vah']:.4f} = zona acceptance -- harga di luar area "
                    f"ini menandakan potensi rejection/ekstensi (bukan skor tambahan)")

    # 11) MM inventory (PROXY) -- sintesis funding+OI+CVD yang SUDAH dihitung
    #     di atas (poin 5-7), BUKAN skor baru (funding/OI/CVD sudah observable
    #     riil, ini cuma reprocessing jadi satu narasi -- MM inventory asli
    #     tidak observable dari data publik).
    if funding and oi and oi.get("trend") is not None:
        rate = funding["rate"]
        bias_txt = "short" if rate > 0 else ("long" if rate < 0 else "netral")
        reasons.append(f"[MM inventory - PROXY] funding {rate:+.4%} + OI trend "
                        f"{oi['trend']:+.2f} + CVD trend {cvd_trend:+.2f} -> estimasi kasar "
                        f"dealer/MM condong sisi {bias_txt} (sintesis 3 faktor di atas, "
                        f"bukan data posisi riil, tidak menambah skor)")

    return dict(
        score=score, reasons=reasons, price=price,
        structure=structure, pools=pools, fvgs=fvgs, vp=vp, vwap=vwap_val,
        delta=delta_now, cvd_trend=cvd_trend, funding=funding, oi=oi, magnet=magnet,
        swing_h=swing_h, swing_l=swing_l,
        order_blocks=order_blocks, structure_source="native",
        trapped=trapped, voids=voids,
    )


# =============================================================================
# 6. SCORING ENGINE — XAU (via PAXGUSDT, playbook hibrida)
# =============================================================================
def analyze_xau(df: pd.DataFrame, symbol: str, has_futures: bool, dxy_bias_override=None,
                 df_structure=None, structure_source="native") -> dict:
    """
    df            : candle PAXGUSDT (proxy) -- dipakai untuk volume profile,
                    VWAP sesi, dan reaksi candle/vol_ratio (SOP: volume real
                    cuma tersedia dari sini).
    df_structure  : candle XAUUSD asli dari TwelveData (via get_structure_df)
                    -- dipakai KHUSUS untuk swing/BOS/liquidity pool/Order
                    Block. Kalau None, fallback ke df (perilaku lama, pra-hybrid).
    structure_source : label asal df_structure ("twelvedata" / "paxg_fallback_*"
                    / "native") supaya laporan transparan soal skala harga yang
                    dipakai (lihat SOP Opsi A -- label, jangan disamarkan).
    """
    price = float(df["close"].iloc[-1])
    struct_df = df_structure if df_structure is not None else df
    # Harga penutupan terakhir DALAM SKALA struct_df (sama dengan skala semua
    # level OB/pool/swing di bawah). Kalau struct_df == df (native/fallback),
    # ini identik dengan `price` (offset = 0). Kalau struct_df = TwelveData
    # XAUUSD asli sementara df = proxy PAXGUSDT, dua angka ini BISA beda
    # $1-5 -- build_setup() WAJIB pakai struct_price ini (bukan `price`)
    # untuk membandingkan/memilih level yang berasal dari struct_df, supaya
    # tidak salah pilih OB atau salah hitung jarak SL akibat skala campur.
    struct_price = float(struct_df["close"].iloc[-1])
    swing_h, swing_l = find_swings(struct_df, left=3, right=3)
    structure = classify_structure(struct_df, swing_h, swing_l)
    pools = find_liquidity_pools(swing_h, swing_l)
    pools = mark_swept_pools(struct_df, pools)
    order_blocks = find_order_blocks(struct_df, structure.get("bos"), structure.get("bos_idx"))
    order_blocks = mark_mitigated_ob(struct_df, order_blocks)
    vp = volume_profile(df.tail(150))

    session_df, session_start = session_window_today(
        df, LONDON_NY_OVERLAP_START_UTC, LONDON_NY_OVERLAP_END_UTC)
    vwap_val, _ = anchored_vwap(df, anchor_ts=session_start)

    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    upper_wick = float(last["high"] - max(last["close"], last["open"]))
    lower_wick = float(min(last["close"], last["open"]) - last["low"])
    body = float(abs(last["close"] - last["open"]))
    avg_vol = float(df["volume"].tail(20).mean())
    vol_ratio = float(last["volume"] / avg_vol) if avg_vol > 0 else 1.0
    if rng > 0 and lower_wick / rng > 0.5 and vol_ratio > 1.2:
        reaction = "bullish_rejection"
    elif rng > 0 and upper_wick / rng > 0.5 and vol_ratio > 1.2:
        reaction = "bearish_rejection"
    elif rng > 0 and body / rng > 0.6 and vol_ratio > 1.1:
        reaction = "continuation_bull" if last["close"] > last["open"] else "continuation_bear"
    else:
        reaction = "neutral"

    dxy_info = None if dxy_bias_override else fetch_dxy_proxy()

    funding = fetch_funding_rate(symbol) if has_futures else None
    oi = fetch_open_interest(symbol) if has_futures else None
    magnet = liquidation_magnet_proxy(price, pools, funding, oi) if has_futures else None
    trapped = detect_trapped_traders(struct_df, pools)
    voids = find_liquidity_voids(price, vp)

    score = 0.0
    reasons = []

    # 1) Struktur HTF
    if structure["bias"] == "bullish":
        score += 25
    elif structure["bias"] == "bearish":
        score -= 25
    reasons.append(f"Struktur: {structure['detail']}")
    if structure["bos"]:
        side, lvl = structure["bos"]
        score += 10 if side == "bullish" else -10
        reasons.append(f"Break of Structure {side} menembus level {lvl:.4f}")
    if order_blocks:
        ob0 = order_blocks[0]
        ob_status = "sudah dimitigasi" if ob0.get("mitigated") else "fresh, belum dimitigasi"
        reasons.append(f"Order Block {ob0['type']} @ {ob0['low']:.4f}-{ob0['high']:.4f} ({ob_status}) "
                        f"[sumber struktur: {structure_source}] "
                        f"-- info struktural umum, BELUM TENTU dipakai untuk entry; "
                        f"cek baris 'Order Block dipakai' di atas (kosong = tidak dipakai/FALLBACK)")

    # 2) VWAP sesi London-NY overlap
    if price > vwap_val:
        score += 20
        reasons.append(f"Harga di atas session VWAP London-NY overlap ({vwap_val:.4f})")
    else:
        score -= 20
        reasons.append(f"Harga di bawah session VWAP London-NY overlap ({vwap_val:.4f})")

    # 3) Volume Profile / POC
    if price > vp["vah"]:
        score += 12
        reasons.append(f"Harga di atas Value Area High ({vp['vah']:.4f})")
    elif price < vp["val"]:
        score -= 12
        reasons.append(f"Harga di bawah Value Area Low ({vp['val']:.4f})")
    elif price > vp["poc"]:
        score += 6
        reasons.append(f"Harga di atas POC ({vp['poc']:.4f})")
    else:
        score -= 6
        reasons.append(f"Harga di bawah POC ({vp['poc']:.4f})")

    # 4) Reaksi candle + volume (pengganti order flow real-time)
    reaction_map = {
        "bullish_rejection": (18, "Rejection bullish (lower wick besar + volume tinggi) di area ini"),
        "bearish_rejection": (-18, "Rejection bearish (upper wick besar + volume tinggi) di area ini"),
        "continuation_bull": (12, "Candle body dominan bullish + volume tinggi -> continuation"),
        "continuation_bear": (-12, "Candle body dominan bearish + volume tinggi -> continuation"),
        "neutral": (0, "Reaksi candle netral, belum ada sinyal kuat"),
    }
    delta_score, reaction_reason = reaction_map[reaction]
    score += delta_score
    reasons.append(f"{reaction_reason} (volume {vol_ratio:.2f}x rata-rata 20 bar)")

    # 5) Macro DXY (proxy) — inverse correlation ke gold
    if dxy_bias_override:
        bias = dxy_bias_override.lower()
        if bias == "bearish":  # dolar lemah -> gold cenderung tahan/naik
            score += 20
            reasons.append("Macro override: DXY bias bearish (dolar lemah) -> mendukung gold")
        elif bias == "bullish":
            score -= 20
            reasons.append("Macro override: DXY bias bullish (dolar kuat) -> menekan gold")
        else:
            reasons.append("Macro override: DXY netral")
    elif dxy_info and dxy_info.get("change_pct") is not None:
        chg = dxy_info["change_pct"]
        if chg < -0.3:
            score += 15
            reasons.append(f"DXY proxy turun {chg:.2f}% ({trend_days_label()}) -> dolar melemah, mendukung gold")
        elif chg > 0.3:
            score -= 15
            reasons.append(f"DXY proxy naik {chg:.2f}% ({trend_days_label()}) -> dolar menguat, menekan gold")
        else:
            reasons.append(f"DXY proxy relatif flat ({chg:+.2f}%) -> makro netral")
        reasons.append(f"[Catatan] DXY proxy = {dxy_info['dxy']:.2f} — {dxy_info['note']}")
    else:
        reasons.append("DXY proxy: data tidak tersedia (gagal fetch / gunakan --dxy-bias manual)")

    # 6) Funding/OI/liquidation magnet dari PAXGUSDT (kalau ada kontrak futures-nya)
    if has_futures and funding:
        rate = funding["rate"]
        if rate > FUNDING_EXTREME:
            score -= 10
            reasons.append(f"[PAXG futures] Funding {rate:+.4%} -> long crowded di sisi crypto-nya")
        elif rate < -FUNDING_EXTREME:
            score += 10
            reasons.append(f"[PAXG futures] Funding {rate:+.4%} -> short crowded di sisi crypto-nya")
        if magnet:
            reasons.append(f"[PAXG futures] Liquidation magnet proxy {magnet['direction']} harga @ "
                            f"{magnet['price']:.4f} (pool {magnet['pool_side']}) — "
                            + "; ".join(magnet["notes"]))
    else:
        reasons.append("Funding/OI/liquidation magnet PAXGUSDT: kontrak futures tidak terdeteksi, "
                        "bagian ini dilewati (bobot dialihkan ke struktur/VWAP/macro)")

    # 7) Trapped buyers/sellers (fitur baru, lihat detect_trapped_traders())
    for ev in trapped:
        if ev["kind"] == "trapped_buyers":
            score -= 10
            reasons.append(f"Trapped buyers: buyside liquidity @ {ev['pool_price']:.4f} disapu "
                            f"({ev['touches']}x equal high) lalu harga reverse turun (close "
                            f"{ev['confirm_close']:.4f}) -> potensi bahan bakar tekanan jual lanjutan")
        else:
            score += 10
            reasons.append(f"Trapped sellers: sellside liquidity @ {ev['pool_price']:.4f} disapu "
                            f"({ev['touches']}x equal low) lalu harga reverse naik (close "
                            f"{ev['confirm_close']:.4f}) -> potensi bahan bakar tekanan beli lanjutan")

    # 8) Liquidity void (LVN dari volume profile) -- INFORMASIONAL, TIDAK
    #    menambah skor (sudah dihitung via POC/VAH/VAL di poin 3).
    for v in voids[:2]:
        reasons.append(f"Liquidity void (LVN) {v['side']} harga di {v['price_low']:.4f}-"
                        f"{v['price_high']:.4f} (volume tipis di histori) -> kandidat target "
                        f"kalau harga bergerak ke arah situ, cenderung dilewati cepat")

    # 9) AMT (Auction Market Theory) -- INTERPRETASI di atas Volume Profile
    #    yang sudah dihitung (poin 3), BUKAN faktor skor terpisah.
    reasons.append(f"[AMT] POC {vp['poc']:.4f} = fair value price; area value "
                    f"{vp['val']:.4f}-{vp['vah']:.4f} = zona acceptance -- harga di luar area "
                    f"ini menandakan potensi rejection/ekstensi (bukan skor tambahan)")

    # 10) MM inventory (PROXY) -- sintesis funding PAXG yang sudah dihitung
    #     di atas (poin 6), BUKAN skor baru. XAU tidak punya CVD/OI crypto asli
    #     jadi proxy-nya lebih tipis daripada versi crypto -- ditandai jelas.
    if has_futures and funding:
        rate = funding["rate"]
        bias_txt = "short" if rate > 0 else ("long" if rate < 0 else "netral")
        reasons.append(f"[MM inventory - PROXY, tipis] [PAXG futures] funding {rate:+.4%} -> "
                        f"estimasi sangat kasar dealer/MM sisi crypto proxy condong {bias_txt} "
                        f"(cuma 1 faktor, bukan data posisi riil XAUUSD, tidak menambah skor)")

    return dict(
        score=score, reasons=reasons, price=price, struct_price=struct_price,
        structure=structure, pools=pools, vp=vp, vwap=vwap_val,
        reaction=reaction, vol_ratio=vol_ratio, dxy=dxy_info,
        dxy_override=dxy_bias_override, funding=funding, oi=oi, magnet=magnet,
        swing_h=swing_h, swing_l=swing_l, session_start=session_start,
        order_blocks=order_blocks, structure_source=structure_source,
        trapped=trapped, voids=voids,
    )


def trend_days_label():
    return "5 hari terakhir"


# =============================================================================
# 7. SETUP BUILDER — SL/TP berbasis STRUKTUR (bukan lagi ATR multiplier)
# =============================================================================
def build_setup(a: dict, rr: float = 2.0) -> dict:
    price = a["price"]
    pools = a["pools"]
    vp = a["vp"]
    swing_h, swing_l = a["swing_h"], a["swing_l"]
    order_blocks = a.get("order_blocks", [])

    # struct_price = harga dalam skala yang SAMA dengan pools/swing_h/swing_l/
    # order_blocks (lihat analyze_xau). Untuk asset_class crypto (analyze_crypto)
    # field ini tidak ada -> fallback ke price, offset = 0, perilaku tidak
    # berubah. scale_offset dipakai HANYA di jalur fallback (swing_fallback_
    # no_ob) untuk menggeser level struct-scale ke skala `price` sebelum
    # dipakai menghitung jarak SL dari entry=price -- lihat komentar di
    # pick_ob()/fallback branch di bawah untuk kenapa ini perlu.
    struct_price = a.get("struct_price", price)
    scale_offset = price - struct_price

    SCORE_MAX = 100.0
    conf = 50.0 + 40.0 * min(1.0, abs(a["score"]) / SCORE_MAX)

    if a["score"] > NEUTRAL_THRESHOLD:
        direction = "LONG"
    elif a["score"] < -NEUTRAL_THRESHOLD:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    def nearest_below(val):
        cands = [p["price"] for p in pools if p["price"] < val]
        if swing_l:
            cands += [s[2] for s in swing_l if s[2] < val]
        return max(cands) if cands else None

    def nearest_above(val):
        cands = [p["price"] for p in pools if p["price"] > val]
        if swing_h:
            cands += [s[2] for s in swing_h if s[2] > val]
        return min(cands) if cands else None

    def targets_above(val, tol_ratio=0.0015, ref_offset=0.0):
        """Semua level pool/swing di atas val, urut terdekat->terjauh,
        level yang berhimpitan (<tol_ratio dari val) digabung jadi satu.

        ref_offset (fix price-scale mismatch): pools/swing_h ada dalam skala
        struct_price, sementara `val` (entry) bisa dalam skala `price` proxy
        kalau dipanggil dari jalur fallback (swing_fallback_no_ob). ref_offset
        = price - struct_price mengonversi val ke skala struct SEBELUM
        difilter, lalu hasilnya digeser balik ke skala val supaya konsisten
        dengan entry yang dipakai caller. Default 0.0 = tidak ada konversi
        (dipakai saat val sudah dalam skala struct, mis. entry dari edge OB).
        """
        struct_val = val - ref_offset
        cands = [p["price"] for p in pools if p["price"] > struct_val]
        if swing_h:
            cands += [s[2] for s in swing_h if s[2] > struct_val]
        cands = sorted(set(cands))
        merged = []
        for c in cands:
            c = c + ref_offset
            if not merged or (c - merged[-1]) > val * tol_ratio:
                merged.append(c)
        return merged

    def targets_below(val, tol_ratio=0.0015, ref_offset=0.0):
        """Sama seperti targets_above tapi ke bawah, urut terdekat->terjauh.
        Lihat catatan ref_offset di targets_above()."""
        struct_val = val - ref_offset
        cands = [p["price"] for p in pools if p["price"] < struct_val]
        if swing_l:
            cands += [s[2] for s in swing_l if s[2] < struct_val]
        cands = sorted(set(cands), reverse=True)
        merged = []
        for c in cands:
            c = c + ref_offset
            if not merged or (merged[-1] - c) > val * tol_ratio:
                merged.append(c)
        return merged

    # Jarak SL akhir (dari entry) TIDAK BOLEH lebih kecil dari ini, apa pun
    # posisi liquidity pool/swing/OB terdekat.
    MIN_SL_PCT = 0.001  # 0.1% dari harga; naikkan kalau masih terasa terlalu ketat

    def pick_ob(dir_):
        """
        OB tervalid (belum mitigated) TERDEKAT ke harga, searah trade -- sesuai
        SOP. Untuk LONG: OB bullish yang high-nya <= harga sekarang (OB ada DI
        BAWAH harga, karena impuls yang memicu BOS sudah membawa harga naik
        menjauhinya) -> pilih yang high-nya paling tinggi (paling dekat ke
        harga). Untuk SHORT: simetris terbalik.

        PENTING (fix price-scale mismatch): perbandingan di bawah memakai
        `struct_price`, BUKAN `price`. order_blocks berasal dari struct_df
        (bisa XAUUSD asli/TwelveData), sedangkan `price` ada di skala proxy
        PAXGUSDT/live. Membandingkan o["high"]/o["low"] (skala struct_df)
        terhadap `price` (skala proxy) bisa salah menerima/menolak OB kalau
        dua skala itu selisih beberapa dollar -- entry/SL yang dihasilkan
        (ob_used["high"]/["low"]) TETAP dalam skala struct_df aslinya, jadi
        tidak perlu dikonversi lagi setelah OB terpilih benar.
        """
        want_type = "bullish" if dir_ == "LONG" else "bearish"
        cands = [o for o in order_blocks if o["type"] == want_type and not o.get("mitigated", False)]
        if not cands:
            return None
        if dir_ == "LONG":
            cands = [o for o in cands if o["high"] <= struct_price]
            return max(cands, key=lambda o: o["high"]) if cands else None
        else:
            cands = [o for o in cands if o["low"] >= struct_price]
            return min(cands, key=lambda o: o["low"]) if cands else None

    if direction == "NEUTRAL":
        entry = price
        sl = tp1 = tp2 = tp3 = price
        risk = 0.0
        order_type = None
        entry_basis = None
        ob_used = None
        tier = None
    elif direction == "LONG":
        ob_used = pick_ob("LONG")
        if ob_used is not None:
            # Entry = edge OB (edge ATAS OB bullish). SL = edge BAWAH OB + buffer.
            entry = ob_used["high"]
            buffer_ = entry * OB_BUFFER_PCT
            raw_dist = entry - ob_used["low"] + buffer_
            min_dist = entry * MIN_SL_PCT
            sl_dist_final = max(raw_dist, min_dist)
            sl = entry - sl_dist_final
            entry_basis = "edge_ob"
        else:
            # Fallback: tidak ada OB valid searah trade -> logika lama
            # (nearest swing/pool + buffer 10% dari jarak).
            # Fix price-scale mismatch: pilih level dalam skala struct_price
            # (skala yang sama dengan pools/swing_l), lalu geser balik ke
            # skala `price` (+scale_offset) sebelum dipakai hitung jarak dari
            # entry=price -- kalau tidak, raw_dist bisa keliru sebesar selisih
            # skala proxy vs struktur asli (bisa $1-5 di XAUUSD).
            sl_level = nearest_below(struct_price)
            if sl_level is None:
                sl_level = price * 0.985
            else:
                sl_level = sl_level + scale_offset
            raw_dist = price - sl_level
            min_dist = price * MIN_SL_PCT
            dist = max(raw_dist, min_dist)
            buffer_ = dist * 0.10
            entry = price
            sl = entry - dist - buffer_
            sl_dist_final = entry - sl
            entry_basis = "swing_fallback_no_ob"

        # TP1/TP2/TP3 = pool/swing ke-1/2/3 terdekat searah trade dari ENTRY
        # (logika liquidity-based lama, tidak berubah -- cuma basisnya
        # sekarang `entry`, bukan `price`, karena entry bisa berbeda dari
        # harga sekarang begitu basisnya Order Block).
        # ref_offset=0 di jalur edge_ob (entry sudah skala struct); di jalur
        # fallback entry=price (skala proxy) -> perlu scale_offset supaya
        # target pool/swing (skala struct) dibandingkan & dikembalikan benar.
        cand_targets = targets_above(entry, ref_offset=(0.0 if entry_basis == "edge_ob" else scale_offset))
        min_gap = sl_dist_final * 0.3

        tp1 = (cand_targets[0] if len(cand_targets) >= 1
               and cand_targets[0] - entry >= min_gap
               else entry + sl_dist_final * 1.0)
        r1_reached = (tp1 - entry) / sl_dist_final if sl_dist_final > 0 else 1.0
        tp2 = (cand_targets[1] if len(cand_targets) >= 2
               and cand_targets[1] > tp1
               else entry + sl_dist_final * max(rr, r1_reached + 1.0))
        r2_reached = (tp2 - entry) / sl_dist_final if sl_dist_final > 0 else 1.0
        tp3 = (cand_targets[2] if len(cand_targets) >= 3
               and cand_targets[2] > tp2
               else entry + sl_dist_final * max(rr * 1.5, r2_reached + 1.0))
        tp2 = max(tp2, tp1 + sl_dist_final * 0.1)
        tp3 = max(tp3, tp2 + sl_dist_final * 0.1)

        risk = entry - sl
        order_type = "LIMIT/MARKET (struktur bullish)"
        tier = classify_tier(risk, abs(tp1 - entry), abs(tp2 - entry))
    else:  # SHORT
        ob_used = pick_ob("SHORT")
        if ob_used is not None:
            entry = ob_used["low"]
            buffer_ = entry * OB_BUFFER_PCT
            raw_dist = ob_used["high"] - entry + buffer_
            min_dist = entry * MIN_SL_PCT
            sl_dist_final = max(raw_dist, min_dist)
            sl = entry + sl_dist_final
            entry_basis = "edge_ob"
        else:
            # Sama seperti fallback LONG di atas -- pilih di skala struct_price,
            # geser balik ke skala `price` sebelum dipakai hitung jarak.
            sl_level = nearest_above(struct_price)
            if sl_level is None:
                sl_level = price * 1.015
            else:
                sl_level = sl_level + scale_offset
            raw_dist = sl_level - price
            min_dist = price * MIN_SL_PCT
            dist = max(raw_dist, min_dist)
            buffer_ = dist * 0.10
            entry = price
            sl = entry + dist + buffer_
            sl_dist_final = sl - entry
            entry_basis = "swing_fallback_no_ob"

        cand_targets = targets_below(entry, ref_offset=(0.0 if entry_basis == "edge_ob" else scale_offset))
        min_gap = sl_dist_final * 0.3

        tp1 = (cand_targets[0] if len(cand_targets) >= 1
               and entry - cand_targets[0] >= min_gap
               else entry - sl_dist_final * 1.0)
        r1_reached = (entry - tp1) / sl_dist_final if sl_dist_final > 0 else 1.0
        tp2 = (cand_targets[1] if len(cand_targets) >= 2
               and cand_targets[1] < tp1
               else entry - sl_dist_final * max(rr, r1_reached + 1.0))
        r2_reached = (entry - tp2) / sl_dist_final if sl_dist_final > 0 else 1.0
        tp3 = (cand_targets[2] if len(cand_targets) >= 3
               and cand_targets[2] < tp2
               else entry - sl_dist_final * max(rr * 1.5, r2_reached + 1.0))
        tp2 = min(tp2, tp1 - sl_dist_final * 0.1)
        tp3 = min(tp3, tp2 - sl_dist_final * 0.1)

        risk = sl - entry
        order_type = "LIMIT/MARKET (struktur bearish)"
        tier = classify_tier(risk, abs(tp1 - entry), abs(tp2 - entry))

    return dict(direction=direction, entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                rr=rr, risk=risk, conf=conf, order_type=order_type, price_now=price,
                poc=vp["poc"], vah=vp["vah"], val=vp["val"],
                entry_basis=entry_basis, order_block=ob_used, tier=tier)


# Kombinasi (asset_class, timeframe) yang SUDAH punya bukti EV negatif dari
# sampel besar -> tradeable dipaksa False (laporan tetap menampilkan setup).
# Sumber: backtest_multi.py 2026-09-24, XAUUSD proxy PAXGUSDT, H1 125 hari,
# 165 trade selesai, win rate 40.0%, expectancy -0.08R. Tinjau ulang hanya
# setelah re-test dengan n_decided dan rentang data yang tercatat.
TRADEABLE_BLOCKLIST = {
    ("xau", "H1"): "backtest 2026-09-24: 165 trade H1, win rate 40.0%, expectancy -0.08R (negatif)",
}


def evaluate_tradeable(a: dict, s: dict, order_type: str = None,
                       asset_class: str = None, tf_label: str = None) -> dict:
    """
    Filter KONSERVATIF -- TIDAK menghapus/menyembunyikan setup apa pun dari
    laporan (SOP: semua tier tetap ditampilkan apa adanya). tradeable=True
    HANYA lewat SATU jalur: "fresh_ob_pending" -- Tier A DAN entry dari
    Order Block asli (bukan fallback swing/pool) DAN order type LIMIT
    (BUY LIMIT/SELL LIMIT, pending -- tidak mengejar harga sekarang).

    KENAPA "bias HTF sudah confirmed" (3 swing bersih searah) TIDAK LAGI
    dipakai sebagai syarat tradeable=True SENDIRIAN, walau dulu jadi jalur
    utama: dari SEMUA backtest sejauh ini (termasuk yang di bawah), bias
    "confirmed" konsisten nyaris sama dengan baseline tanpa filter -- tidak
    terbukti memberi edge tambahan. Itu bagian yang masih valid.

    [KOREKSI 2026-09-24] Klaim lama di sini ("fresh_ob_pending win rate
    90-97% di XAUUSD, tiga timeframe") DICABUT -- TIDAK terbukti reproduce
    dan sudah TERBANTAHKAN oleh backtest dengan sampel jauh lebih besar:

        backtest_multi.py, 2026-09-24, XAUUSD (proxy PAXGUSDT),
        jalur TRADEABLE_TRUE ("fresh_ob_pending"):
            M15 (41 hari, ~4000 candle) : win rate 69.8%  (159 trade selesai)
            H1  (125 hari, ~3000 candle): win rate 40.0%  (165 trade selesai,
                                          expectancy NEGATIF -0.08R)
            H4  (333 hari, ~2000 candle): win rate 48.1%  (77 trade selesai)
        Kandidat TERBAIK di run itu (n>=15 trade, semua 6 kombinasi symbol x
        timeframe yang diuji) justru BTCUSDT H4 (+0.73R, win rate 61.9%),
        BUKAN XAUUSD sama sekali.

    Klaim 90-97% yang lama kemungkinan besar tidak valid karena salah satu
    atau kombinasi dari: (a) dihitung dari sampel yang jauh lebih kecil
    (rentang waktu tidak sejelas run 2026-09-24 di atas), dan/atau (b) state
    kode saat itu belum bersih -- lihat riwayat OB_IMPULSE_BODY_MULT di atas
    modul ini: parameter itu sempat dikalibrasi SAAT bos_idx masih bug
    (selalu = candle terakhir, bukan candle breakout asli), jadi backtest
    yang jalan di sekitar tanggal itu tidak otomatis bisa dipercaya tanpa
    tahu persis versi kode & jumlah trade di baliknya. Log/command asli yang
    menghasilkan klaim 90-97% itu TIDAK tersedia lagi untuk ditelusuri ulang.

    [UPDATE 2026-09-24 -- RUN ULANG, RENTANG LEBIH PANJANG] jalur TRADEABLE_TRUE
    ("fresh_ob_pending"), struktur proxy PAXGUSDT utk XAUUSD. Command:
        python -X utf8 backtest_multi.py --xau-tf M15 --btc-tf H4 --limit 9000 --step 3 --out-dir results_run2_m15
        python -X utf8 backtest_multi.py --xau-tf H4 --btc-tf H4 --limit 3000 --out-dir results_run2_h4
    Hasil (n = trade SELESAI, rr=2.0, warmup 200, SL duluan kalau ambigu):
        XAUUSD M15 : 2026-06-22..2026-09-23 (~94 hari, step 3) -> win 60.4%, +0.29R, n=106
        XAUUSD H4  : 2025-05-11..2026-09-23 (step 1)           -> win 48.2%, +0.35R, n=137
        BTCUSDT H4 : 2022-08-15..2026-09-23 (step 3)           -> win 63.9%, +0.68R, n=133
        BTCUSDT H4 : 2025-05-11..2026-09-23 (step 1)           -> win 61.2%, +0.66R, n=165
    Pembanding tanpa filter (BASELINE): XAUUSD M15 +0.10R, XAUUSD H4 -0.10R,
    BTCUSDT H4 -0.13R (2022-2026) / +0.01R (2025-2026).

    KESIMPULAN SEMENTARA (2026-09-24): jalur "fresh_ob_pending" punya expectancy
    positif konsisten di BTCUSDT H4 (3 run terpisah, +0.66R..+0.73R), dan run
    ulang juga positif di XAUUSD H4 (+0.35R, win rate 48.2% sama dgn run
    sebelumnya) dan XAUUSD M15 (+0.29R, turun dari win 69.8% di sampel 41 hari
    -> 60.4% di 94 hari, wajar utk regresi ke rata-rata). XAUUSD H1 tetap
    DIBLOKIR (TRADEABLE_BLOCKLIST) krn -0.08R dari 165 trade -- CATATAN: H1
    baru diuji SEKALI (125 hari), belum diulang di rentang lebih panjang.
    [ANALISIS TRADE LOG 2026-09-24, trades_*_H4.csv, 2025-05..2026-09, step 1]
    Trade yang selesai ternyata banyak DUPLIKAT dari setup yg sama (kunci =
    arah+entry+SL): BTCUSDT H4 165 trade = hanya 51 setup unik -> win 58.8%,
    +0.56R, bootstrap 95% CI [+0.16, +0.98]. XAUUSD H4 137 trade = hanya 45
    setup unik -> win 46.7%, +0.16R, 95% CI [-0.22, +0.55] (TIDAK bisa
    dibedakan dari nol). Ada juga kuartal negatif: BTC 2026Q3 -0.35R, XAU
    2026Q1 -0.67R; hasil BTC sangat ditopang 2025Q4 (+1.91R). 43% (BTC) dan
    61% (XAU) sinyal tradeable tidak pernah selesai (FILL_TIMEOUT).
    XAUUSD M15 (trades_XAUUSD_M15.csv, 2026-06-24..2026-09-23, step 3): 106
    trade = 77 setup unik -> win 58.4%, +0.25R, 95% CI [+0.01, +0.49]
    (batas bawah nyaris nol). Expectancy per bulan menurun: Jul +0.45R,
    Ags +0.30R, Sep +0.05R. 66% sinyal tradeable FILL_TIMEOUT. Median jarak
    SL hanya ~$7.8 (spread/slippage broker memakan porsi R yg berarti).
    => EVIDENCE: BTCUSDT H4 cukup kuat; XAUUSD H4/M15 LEMAH, jangan diperlakukan
    sebagai edge terbukti.

    KETERBATASAN: (1) n trade BERKORELASI (duplikat setup, lihat analisis di
    atas), sampel efektif jauh lebih kecil dari n; (2) filter/parameter dikalibrasi dari data yg sebagian sama --
    bukan out-of-sample murni; (3) proxy PAXGUSDT, tanpa spread/slippage/
    komisi broker (paling berpengaruh di M15); (4) tradeable=True tetap
    "lolos filter konservatif", BUKAN jaminan probabilitas.

    ATURAN UNTUK KOMENTAR/DOCSTRING WIN-RATE DI FILE INI KE DEPAN: setiap
    klaim angka win rate/expectancy WAJIB mencantumkan (1) tanggal run,
    (2) skrip & command persis yang dipakai, (3) jumlah trade SELESAI
    (n_decided) per kombinasi, (4) simbol & rentang tanggal data. Klaim
    tanpa keempat hal itu tidak boleh dipakai sebagai dasar keputusan filter
    -- itu persis yang membuat klaim 90-97% di atas jadi tidak bisa
    diverifikasi ulang.

    CATATAN: kesimpulan di atas dari backtest pakai struktur proxy PAXGUSDT
    (BUKAN histori XAUUSD asli), tanpa slippage/komisi/spread broker riil,
    bukan jaminan berlaku selamanya/di semua kondisi. Kalibrasi ulang kalau
    ada bukti baru -- tapi WAJIB sampling ulang, jangan asumsikan hasil lama
    (termasuk kesimpulan sementara di atas) otomatis masih benar.
    """
    if s.get("direction") == "NEUTRAL":
        return {"tradeable": False, "path": None,
                "reason": "Tidak ada sinyal arah (skor di zona NEUTRAL, tidak ada setup)."}

    block_reason = TRADEABLE_BLOCKLIST.get((asset_class, (tf_label or "").upper()))
    if block_reason:
        return {"tradeable": False, "path": None,
                "reason": f"[BLOCKED] Kombinasi {asset_class}/{tf_label} diblokir dari "
                           f"tradeable=True karena bukti EV negatif ({block_reason}). "
                           f"Setup tetap ditampilkan apa adanya di laporan."}

    # Sanity-check skala harga (jaring pengaman untuk bug price-scale
    # mismatch, lihat build_setup()/pick_ob()) -- kalau structure_source
    # bukan "native"/fallback-tanpa-key (artinya struct_df punya sumber
    # harga TERPISAH dari df proxy) dan selisih price vs struct_price di
    # luar rentang wajar, JANGAN percaya tradeable=True: level entry/SL/OB
    # yang dipilih berpotensi tercemar skala. Ini juga jaring pengaman kalau
    # nanti sumber struktur lain (bukan TwelveData) ditambahkan dengan basis
    # harga yang lebih jauh berbeda lagi.
    price_now = a.get("price")
    struct_price = a.get("struct_price")
    src = a.get("structure_source", "native")
    if (price_now is not None and struct_price is not None and price_now != 0
            and src not in ("native",) and not src.startswith("paxg_fallback")):
        scale_gap_pct = abs(price_now - struct_price) / price_now * 100
        if scale_gap_pct > 0.5:
            return {"tradeable": False, "path": None,
                    "reason": f"[SAFETY] Selisih skala harga proxy vs struktur ({src}) "
                               f"{scale_gap_pct:.2f}% -- di luar batas wajar (>0.5%). "
                               f"Entry/SL/OB berpotensi tidak sinkron skala, tradeable "
                               f"dipaksa False sampai ini dicek manual."}

    tier_ok = s.get("tier") == "A"
    bias = a.get("structure", {}).get("bias")
    bias_confirmed = bias in ("bullish", "bearish")

    has_fresh_ob = s.get("entry_basis") == "edge_ob" and s.get("order_block") is not None
    is_pending_limit = order_type in ("BUY LIMIT", "SELL LIMIT")

    if not tier_ok:
        return {"tradeable": False, "path": None,
                "reason": f"Tier {s.get('tier')} (bukan A) -- risk belum tertutup penuh oleh TP1."}

    if has_fresh_ob and is_pending_limit:
        bias_note = (f"struktur HTF '{bias}' sudah confirmed juga (bonus konteks)"
                      if bias_confirmed else
                      f"struktur HTF '{bias}' belum confirmed, tapi bias HTF tidak terbukti "
                      f"signifikan di backtest manapun sejauh ini")
        return {"tradeable": True, "path": "fresh_ob_pending",
                "reason": f"Tier A, entry dari Order Block asli (BOS), order {order_type} "
                           f"(pending, tidak mengejar harga) -- {bias_note}. CATATAN: lolos "
                           f"filter konservatif, BUKAN jaminan win rate tinggi -- lihat "
                           f"docstring evaluate_tradeable() untuk hasil backtest terbaru "
                           f"per simbol/timeframe sebelum eksekusi."}

    reasons = []
    if not has_fresh_ob:
        reasons.append("Entry bukan dari Order Block asli (fallback swing/liquidity pool).")
    if not is_pending_limit:
        reasons.append(f"Order type '{order_type}' bukan LIMIT pending (mengejar harga sekarang).")
    reasons.append(f"[info] Bias HTF '{bias}' "
                    f"({'sudah confirmed' if bias_confirmed else 'belum confirmed/ranging'}) -- "
                    f"TIDAK dipakai lagi sebagai penentu tradeable (backtest: tidak terbukti unggul "
                    f"dari baseline).")
    return {"tradeable": False, "path": None, "reason": "; ".join(reasons)}


# =============================================================================
# 8. REPORT PRINTER
# =============================================================================
def fmt(x, dp=4):
    return f"{x:.{dp}f}"


def classify_order_type(direction: str, entry: float, ref_price, tol_pct: float = 0.0008):
    """Tentukan tipe order (LIMIT/STOP/MARKET) dengan membandingkan level
    ENTRY (dihitung dari harga proxy PAXGUSDT/gold-api) terhadap harga
    referensi REAL (spot XAUUSD dari fetch_spot_crosscheck, yang jauh lebih
    dekat ke harga broker seperti icMarkets daripada proxy).

    Kenapa ini perlu: entry selalu dihitung dari `price` (proxy), yang bisa
    selisih beberapa dollar dari harga broker user. Sebelumnya order_type
    cuma teks generik "LIMIT/MARKET" -- tidak bilang entry itu pending
    (nunggu harga menjemput) atau sudah bisa market. Sekarang dibandingkan
    langsung ke spot_price supaya jelas.

    Kalau spot_price tidak tersedia (None), kembalikan type=None -- caller
    harus fallback ke label generik dan kasih catatan bahwa order type
    belum bisa dipastikan tanpa harga referensi live.
    """
    if direction not in ("LONG", "SHORT") or not ref_price:
        return {"type": None, "note": None}

    diff = entry - ref_price
    diff_pct = diff / ref_price if ref_price else 0.0

    if abs(diff_pct) <= tol_pct:
        return {
            "type": "MARKET",
            "note": (f"Entry ({fmt(entry)}) hampir sama dengan harga spot referensi "
                     f"({fmt(ref_price)}) -> bisa dieksekusi market, tapi tetap cek "
                     "ulang harga live di broker sebelum entry."),
        }

    if direction == "LONG":
        if entry < ref_price:
            return {
                "type": "BUY LIMIT",
                "note": (f"Entry ({fmt(entry)}) di BAWAH harga spot referensi "
                         f"({fmt(ref_price)}) -> ini pending order, menunggu harga "
                         "turun dulu ke level entry. BUKAN market order."),
            }
        return {
            "type": "BUY STOP",
            "note": (f"Entry ({fmt(entry)}) di ATAS harga spot referensi "
                     f"({fmt(ref_price)}) -> ini pending order, menunggu harga "
                     "tembus naik ke level entry dulu."),
        }
    else:  # SHORT
        if entry > ref_price:
            return {
                "type": "SELL LIMIT",
                "note": (f"Entry ({fmt(entry)}) di ATAS harga spot referensi "
                         f"({fmt(ref_price)}) -> ini pending order, menunggu harga "
                         "naik dulu ke level entry. BUKAN market order."),
            }
        return {
            "type": "SELL STOP",
            "note": (f"Entry ({fmt(entry)}) di BAWAH harga spot referensi "
                     f"({fmt(ref_price)}) -> ini pending order, menunggu harga "
                     "tembus turun ke level entry dulu."),
        }


def select_relevant_pools(pools, price, max_n=4):
    """
    Pilih liquidity pool yang PALING RELEVAN buat ditampilkan: selalu
    utamakan pool unswept terdekat di atas & di bawah harga (ini yang
    dipakai liquidation_magnet_proxy untuk menentukan magnet), baru sisanya
    diisi pool dengan jumlah 'touches' terbanyak. Ini supaya level yang
    disebut di ALASAN/magnet selalu muncul juga di daftar ini (sebelumnya
    daftar cuma top-3 by touches, jadi bisa beda dgn level yang dirujuk
    magnet -> membingungkan).
    """
    if not pools:
        return []
    unswept = [p for p in pools if not p.get("swept", False)]
    above = [p for p in unswept if p["price"] > price]
    below = [p for p in unswept if p["price"] < price]
    nearest_above = min(above, key=lambda p: p["price"] - price) if above else None
    nearest_below = min(below, key=lambda p: price - p["price"]) if below else None

    selected, seen = [], set()
    for p in (nearest_above, nearest_below):
        if p is not None:
            key = round(p["price"], 6)
            if key not in seen:
                selected.append(p)
                seen.add(key)
    remaining = sorted(
        [p for p in pools if round(p["price"], 6) not in seen],
        key=lambda p: p["touches"], reverse=True,
    )
    for p in remaining:
        if len(selected) >= max_n:
            break
        selected.append(p)
        seen.add(round(p["price"], 6))
    return selected[:max_n]


def print_report(display_symbol, binance_symbol, tf_label, a, s, asset_class,
                  is_alias=False, spot_price=None, spot_label=None, cot=None):
    line = "=" * 70
    print(f"\n{line}")
    tag = f"(data via {binance_symbol})" if is_alias else ""
    print(f"  AI TRADING ANALYSIS v2 (Liquidity/Structure-based)  |  {display_symbol}  {tf_label}  {tag}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    (ANALYSIS-ONLY, eksekusi tetap manual)")
    print(line)

    arrow = {"LONG": "▲ LONG (beli)", "SHORT": "▼ SHORT (jual)",
             "NEUTRAL": "= NEUTRAL (belum ada setup jelas)"}[s["direction"]]
    print(f"  Arah        : {arrow}")
    print(f"  Confidence  : {s['conf']:.0f}%   (skor multi-faktor: {a['score']:+.0f})")
    live_tag = "live" if a.get("price_is_live") else "close candle terakhir (live price gagal diambil)"
    print(f"  Harga now   : {fmt(a['price'])}  ({live_tag})")
    if a.get("price_closed_candle") is not None and a.get("price_is_live"):
        print(f"  (Ref: close candle {tf_label} terakhir yang sudah closed = {fmt(a['price_closed_candle'])})")
    if spot_price is not None:
        spread = a["price"] - spot_price
        spread_pct = (spread / spot_price * 100) if spot_price else 0
        print(f"  Spot ref    : {fmt(spot_price)}  ({spot_label})   selisih proxy: {spread:+.2f} ({spread_pct:+.2f}%)")
    print("-" * 70)

    if s["direction"] != "NEUTRAL":
        oc = classify_order_type(s["direction"], s["entry"], spot_price)
        if oc["type"]:
            print(f"  ORDER TYPE  : {oc['type']}")
            print(f"  {'':12}  {oc['note']}")
        else:
            print(f"  ORDER TYPE  : {s['order_type']}  (perkiraan -- harga spot referensi "
                  "tidak tersedia, bandingkan manual dengan harga broker sebelum entry)")
        basis_txt = {
            "edge_ob": "edge Order Block (lihat detail OB di bawah)",
            "swing_fallback_no_ob": "FALLBACK -- tidak ada Order Block valid searah trade, "
                                     "pakai nearest swing/pool + buffer",
        }.get(s.get("entry_basis"), "harga sekarang (proxy)")
        print(f"  ENTRY       : {fmt(s['entry'])}   (basis: {basis_txt})")
        print(f"  STOP LOSS   : {fmt(s['sl'])}   (edge OB berlawanan + buffer {OB_BUFFER_PCT*100:.2f}%, "
              "atau fallback nearest swing/pool kalau tidak ada OB)")
        if s.get("order_block"):
            ob = s["order_block"]
            print(f"   Order Block dipakai : {ob['type']} @ {fmt(ob['low'])}-{fmt(ob['high'])}")
        if s.get("tier"):
            tier_note = {"A": "layak penuh", "B": "layak marjinal (TP1 belum menutup risiko)",
                         "C": "berisiko tinggi (risk > jarak TP2)"}.get(s["tier"], "")
            print(f"  TIER        : {s['tier']}  ({tier_note})")
        if s["risk"] > 0:
            r1 = abs(s["tp1"] - s["entry"]) / s["risk"]
            r2 = abs(s["tp2"] - s["entry"]) / s["risk"]
            r3 = abs(s["tp3"] - s["entry"]) / s["risk"]
        else:
            r1 = r2 = r3 = 0.0
        print(f"  TP1         : {fmt(s['tp1'])}   ({r1:.2f}R)")
        print(f"  TP2         : {fmt(s['tp2'])}   ({r2:.2f}R)")
        print(f"  TP3         : {fmt(s['tp3'])}   ({r3:.2f}R)")
        print("  [TP1-3 diprioritaskan dari liquidity pool/swing terdekat searah trade,")
        print("   urut dari yang paling dekat; kalau struktur di level segitu tidak")
        print("   tersedia/terlalu dekat, dipakai proyeksi R-multiple sebagai gantinya]")
        if s["risk"] > 0:
            rr_actual = abs(s["tp2"] - s["entry"]) / s["risk"]
            print(f"  Risk:Reward : 1 : {rr_actual:.2f}")
    else:
        print(f"  Tidak ada setup jelas. Tunggu konfirmasi (skor di rentang "
              f"-{NEUTRAL_THRESHOLD}..+{NEUTRAL_THRESHOLD}).")
    print("-" * 70)

    src = a.get("structure_source", "native")
    src_label = {
        "twelvedata": "TwelveData (XAUUSD asli)",
        "native": "data candle simbol ini sendiri",
        "paxg_fallback_no_api_key": "FALLBACK PAXGUSDT (TwelveData API key belum di-set)",
        "paxg_fallback_error": "FALLBACK PAXGUSDT (request TwelveData gagal)",
        "paxg_fallback_insufficient_data": "FALLBACK PAXGUSDT (data TwelveData kurang)",
    }.get(src, src)
    print(f"  STRUKTUR & LIKUIDITAS   [sumber: {src_label}]")
    print(f"   Bias HTF     : {a['structure']['bias'].upper()} — {a['structure']['detail']}")
    if a["structure"]["bos"]:
        side, lvl = a["structure"]["bos"]
        print(f"   BOS terakhir : {side} menembus {fmt(lvl)}")
    vp_label = "  [skala harga: PAXGUSDT]" if asset_class == "xau" else ""
    print(f"   POC / VAH / VAL : {fmt(a['vp']['poc'])} / {fmt(a['vp']['vah'])} / {fmt(a['vp']['val'])}{vp_label}")
    print(f"   VWAP{'  (sesi London-NY)' if asset_class=='xau' else ' (rolling)'} : {fmt(a['vwap'])}")
    if a.get("pools"):
        for p in select_relevant_pools(a["pools"], a["price"], max_n=4):
            status = "SUDAH DISAPU" if p.get("swept") else "belum disapu"
            pos = "di atas" if p["price"] > a["price"] else "di bawah"
            print(f"   Liquidity pool ({p['side']}) @ {fmt(p['price'])}  {pos} harga  "
                  f"[{p['touches']}x equal level, {status}]")
    if a.get("magnet"):
        m = a["magnet"]
        print(f"   Liquidation magnet (PROXY) : {m['direction']} harga @ {fmt(m['price'])} "
              f"(pool {m['pool_side']})")
    if a.get("trapped"):
        for ev in a["trapped"]:
            label = "TRAPPED BUYERS" if ev["kind"] == "trapped_buyers" else "TRAPPED SELLERS"
            print(f"   {label} @ {fmt(ev['pool_price'])} ({ev['touches']}x equal level, "
                  f"reverse confirmed close {fmt(ev['confirm_close'])})")
    if a.get("voids"):
        for v in a["voids"][:2]:
            print(f"   Liquidity void (LVN) {v['side']} harga : {fmt(v['price_low'])}-{fmt(v['price_high'])}")

    print("-" * 70)
    if asset_class == "crypto":
        print("  ORDER FLOW / FUNDING / OI")
        print(f"   Delta candle terakhir : {a['delta']:+.2f}    CVD trend (10 bar): {a['cvd_trend']:+.2f}")
        if a.get("funding"):
            print(f"   Funding rate : {a['funding']['rate']:+.4%}")
        else:
            print("   Funding rate : tidak tersedia")
        if a.get("oi") and a["oi"].get("current") is not None:
            trend_txt = f"{a['oi']['trend']:+.2f}" if a["oi"].get("trend") is not None else "n/a"
            print(f"   Open Interest : {a['oi']['current']:.2f}   (tren 24 bar: {trend_txt})")
        else:
            print("   Open Interest : tidak tersedia")
    else:
        print("  REAKSI CANDLE / MACRO")
        print(f"   Reaksi candle terakhir : {a['reaction']}  (volume {a['vol_ratio']:.2f}x avg20)")
        if a.get("dxy_override"):
            print(f"   DXY bias (manual override) : {a['dxy_override']}")
        elif a.get("dxy"):
            print(f"   DXY proxy : {a['dxy']['dxy']:.2f}  "
                  f"(chg {a['dxy']['change_pct']:+.2f}% / {a['dxy']['note']})")
        else:
            print("   DXY proxy : tidak tersedia")
        if a.get("funding"):
            print(f"   [PAXG futures] Funding rate : {a['funding']['rate']:+.4%}")
        if cot:
            print(f"   COT Gold (INFO SAJA, tidak dihitung skor) — laporan {cot['date']}:")
            print(f"     Non-commercial long/short : {cot['noncomm_long']} / {cot['noncomm_short']}")
            print(f"     Commercial long/short     : {cot['comm_long']} / {cot['comm_short']}")

    print("-" * 70)
    print("  ALASAN (reasoning):")
    for r_ in a["reasons"]:
        print(f"   - {r_}")
    print("-" * 70)
    print("  CATATAN RISIKO & LIMITASI DATA:")
    print("   - Ini ANALISIS, bukan saran keuangan. DYOR (Do Your Own Research).")
    print("   - 'Liquidation magnet' adalah PROXY heuristik, BUKAN data notional likuidasi riil.")
    print("   - DXY adalah proxy harian (bukan intraday real-time).")
    print("   - Slippage & fee tidak dihitung; SL bisa tersentuh karena spike/gap.")
    print("   - Confidence tinggi TIDAK menjamin profit. Kelola posisi & ukuran.")
    if is_alias:
        print(f"   - {display_symbol} memakai proxy {binance_symbol}: melacak spot, bukan harga XAUUSD persis.")
    print(line + "\n")


# =============================================================================
# 9. CHART (opsional)
# =============================================================================
def plot_chart(display_symbol, binance_symbol, tf_label, df, a, s, window=120, out_dir=None):
    """
    out_dir : direktori tujuan penyimpanan PNG (absolute path). Kalau None,
              simpan di cwd (perilaku lama, CLI). SENGAJA tidak pakai
              os.chdir() supaya aman dipanggil dari banyak thread sekaligus
              (server.py via FastAPI threadpool) -- os.chdir() mengubah cwd
              milik SELURUH proses, bukan per-thread, jadi request paralel
              bisa saling menimpa cwd satu sama lain dan menyimpan/mencari
              file di direktori yang salah.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.tail(window).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(d["open_time"], d["close"], color="#1f77b4", linewidth=1.3, label="Close")

    ax.axhline(a["vwap"], color="#e377c2", linestyle="-", linewidth=1.2, label="VWAP")
    ax.axhline(a["vp"]["poc"], color="#8c564b", linestyle="--", linewidth=1.1, label="POC")
    ax.axhline(a["vp"]["vah"], color="#8c564b", linestyle=":", linewidth=0.9, alpha=0.7, label="VAH")
    ax.axhline(a["vp"]["val"], color="#8c564b", linestyle=":", linewidth=0.9, alpha=0.7, label="VAL")

    for p in sorted(a.get("pools", []), key=lambda p: p["touches"], reverse=True)[:4]:
        col = "#2ca02c" if p["side"] == "sellside" else "#d62728"
        ax.axhline(p["price"], color=col, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(d["open_time"].iloc[-1], p["price"], f" LP {p['side']} {p['price']:.4f}",
                color=col, fontsize=7, va="center")

    if s["direction"] != "NEUTRAL":
        for name, val, col in [("Entry", s["entry"], "#2ca02c"), ("SL", s["sl"], "#d62728"),
                                ("TP1", s["tp1"], "#ff7f0e"), ("TP2", s["tp2"], "#9467bd")]:
            ax.axhline(val, color=col, linestyle="-.", linewidth=1.0, alpha=0.85)
            ax.text(d["open_time"].iloc[0], val, f" {name} {val:.4f}", color=col, fontsize=7, va="center")

    ax.set_title(f"{display_symbol} {tf_label} — {s['direction']} conf {s['conf']:.0f}% "
                 f"(Liquidity/Structure-based)")
    ax.set_xlabel("Waktu"); ax.set_ylabel("Harga")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    fname = f"chart_{display_symbol}_{tf_label}_v2.png"
    fpath = os.path.join(out_dir, fname) if out_dir else fname
    plt.savefig(fpath, dpi=110)
    plt.close()
    print(f"  [chart disimpan: {fpath}]")
    return fpath


# =============================================================================
# 9b. SCALPING CASCADE (XAUUSD-ONLY) — waterfall confluence 5 sub-metode
# =============================================================================
# SOP scalping terpisah dari engine swing di atas. TIDAK dipakai untuk crypto
# (lihat catatan proyek: metode ini hanya jalan di XAUUSD). Reuse komponen
# struktur/SMC yang sudah ada (find_swings, classify_structure,
# find_liquidity_pools, mark_swept_pools, find_fair_value_gaps,
# find_order_blocks, mark_mitigated_ob, classify_tier) -- TIDAK reimplementasi
# ulang deteksi struktur dasar.
#
# Waterfall (dicoba berurutan, berhenti di match PERTAMA — kualitas tinggi ke
# rendah sesuai urutan diskusi SOP):
#   1. Sweep -> Displacement -> CHoCH/MSS -> entry di edge FVG/OB (LIMIT)
#   2. Retest OB/S&D TANPA syarat sweep baru (MARKET, retest sudah terkonfirmasi)
#   3. BOS continuation + pullback ke fib 50-61.8% (LIMIT)
#   4. Rejection candle/pin bar di key level (MARKET, pola baru closed)
#   5. Failed breakout/failed auction (MARKET, baru closed kembali ke dalam)
#
# Window/recency: hanya candle trigger dalam SCALP_LOOKBACK_CANDLES terakhir
# yang dianggap "segar" (SOP: mode scalping pakai 5 candle terakhir).
#
# SL/TP: BUKAN skala swing. Tiap sub-metode punya basis SL sendiri (sweep
# extreme / OB edge / awal impulse / wick pin bar / breakout extreme + buffer)
# tapi hasil akhirnya di-sanity-check terhadap SCALP_SL_MIN_PIPS/MAX_SL_PIPS
# (rentang "beberapa puluh pips" sesuai SOP) -- kalau risk di luar rentang itu,
# setup DIBUANG (bukan scalping lagi, kemungkinan struktur salah timeframe).
# TP1/2/3 pakai RR tetap (SCALP_RR_TP) dari risk aktual, bukan liquidity pool
# terjauh seperti mode swing -- konsisten dengan filosofi "masuk-keluar cepat".
#
# BELUM DIKALIBRASI: parameter di bawah (body_mult displacement, wick/body
# ratio pin bar, toleransi key level, rentang SL pip) adalah nilai desain
# awal untuk skeleton ini, BELUM divalidasi lewat backtest per sub-metode
# (SOP: backtest wajib per sub-metode sebelum dipakai live -- lihat catatan
# proyek). Jangan anggap ini kalibrasi final.
SCALP_PIP_SIZE = 0.1            # asumsi 1 pip XAUUSD = $0.1 (quoting 2 desimal) -- sesuaikan
                                 # kalau feed broker user pakai konvensi lain
SCALP_SL_MIN_PIPS = 8
SCALP_SL_MAX_PIPS = 40
SCALP_SL_BUFFER_PIPS = 2
SCALP_RR_TP = (1.0, 1.5, 2.0)   # RR tetap TP1/TP2/TP3 dari risk aktual
SCALP_LOOKBACK_CANDLES = 5
SCALP_DISPLACEMENT_BODY_MULT = 1.3
SCALP_PIN_WICK_BODY_RATIO = 2.0
SCALP_KEY_LEVEL_TOL_PCT = 0.0015

# --- Cooldown/dedup sinyal (fix 2026-09-25, lihat analisis backtest) --------
# Backtest declustering (analyze_spread_and_clustering.py) menemukan sub-metode
# tertentu -- terutama "Rejection candle/pin bar" -- menembak ulang sinyal
# SERUPA di harga/waktu yang nyaris sama, candle demi candle, sepanjang satu
# trending move. Di backtest mentah itu terhitung sebagai puluhan ribu
# kemenangan independen, padahal itu 1 kemenangan besar dihitung berkali-kali
# (portfolio expectancy anjlok 0.514R -> 0.146R setelah declustered, lihat
# catatan analisis). Kalau ini tidak difilter DI TITIK SINYAL DIHASILKAN, ini
# bukan cuma bias statistik backtest -- di live ini artinya bot bisa buka
# banyak posisi SEARAH yang sebenarnya satu bet yang sama (leverage
# tersembunyi: begitu market reversal tajam, semua posisi itu kena bareng).
#
# Ambang di bawah SENGAJA disamakan dengan CLUSTER_BAR_GAP_MAX/
# CLUSTER_PIP_GAP_MAX di analyze_spread_and_clustering.py, supaya definisi
# "sinyal independen" konsisten antara apa yang lolos live/backtest dan apa
# yang dihitung sebagai 1 cluster saat analisis. Kalau salah satu diubah,
# ubah juga yang lain.
SCALP_COOLDOWN_BAR_GAP_MAX = 3      # <= 3 candle sejak sinyal terakhir
                                     # (sub_method + direction sama) -> duplikat
SCALP_COOLDOWN_PIP_GAP_MAX = 3.0    # DAN entry beda <= 3 pip -> duplikat

# PERBAIKAN (2026-09-26): ditemukan dari inspeksi hotspot 2025-08-20 --
# cooldown di atas (bar+entry saja) LOLOS untuk kasus "harga choppy di
# sekitar 1 level struktural, entry bergeser >3 pip tiap attempt, tapi SL
# PERSIS SAMA (level referensi identik)". Cek sistematis atas seluruh
# histori (lihat check_same_sl_dedup_gap.py) menunjukkan ini bukan kasus
# langka: 23.94% dari seluruh sinyal berurutan (semua sub-metode) punya
# pola ini -- Failed breakout & Rejection candle di 22-26%. SL identik
# adalah sinyal lebih kuat "level/kejadian yang sama" daripada jarak entry
# semata, jadi ditambahkan sebagai syarat TAMBAHAN: kandidat dianggap
# duplikat (dan ditolak) kalau bar dekat DAN (entry dekat ATAU SL identik).
SCALP_COOLDOWN_SL_GAP_MAX_PIP = 0.5  # SL beda <= ini dianggap "level sama"

# --- Concurrency cap SEARAH (2026-09-26) -------------------------------------
# KENAPA INI BERBEDA DARI COOLDOWN DI ATAS: cooldown di atas cuma mendedup
# sinyal (sub_method + direction) YANG SAMA yang beruntun -- itu tidak
# mencegah, misalnya, "Rejection candle" LONG + "Failed breakout" LONG fire
# nyaris bersamaan (dua sub-metode BEDA, arah SAMA), yang tetap 2 posisi
# searah kalau dieksekusi berbarengan (lihat temuan check_concurrency.py).
#
# TIDAK BISA di-cap otomatis dari sisi df/candle di sini, karena server.py
# adalah API sinyal MURNI (lihat header server.py) -- engine ini TIDAK PERNAH
# tahu posisi mana yang masih terbuka di broker (tidak ada koneksi broker,
# eksekusi dilakukan manual oleh user di MT5). Maka cap ini BUKAN otomatis:
# ia hanya aktif kalau caller (server.py, lewat parameter open_long/open_short
# dari user/Custom GPT) MELAPORKAN sendiri berapa posisi searah yang ia tahu
# sedang terbuka saat ini. open_positions default {} / semua 0 -> cap ini
# efeknya TIDAK AKTIF (backward compatible, termasuk untuk backtest yang
# tidak mensimulasikan status posisi riil sama sekali).
#
# KALIBRASI (2026-09-26): dulu tebakan awal, sekarang sudah divalidasi dari
# data historis (33748 sinyal, setelah fix cooldown SL-based di atas
# menghapus duplikat yang tadinya menginflate concurrency). Hasil
# check_concurrency.py untuk 2 kandidat live (Rejection candle + Failed
# breakout, GABUNGAN searah): level 0-1 mencakup ~91-98% waktu (aman),
# level 2 ~5-6.5% waktu, level 3+ ~2-3% waktu, max historis 7-8 (langka,
# didominasi 1 hotspot 2025-08-20 yang sudah diverifikasi manual --
# choppy whipsaw di 1 level, bukan breakout riil). Cap=2 artinya begitu 2
# posisi searah terbuka (lintas sub-metode), sinyal ke-3 searah diblok --
# ini pas menutup celah "leverage tersembunyi" yang jadi alasan awal cap
# ini dibuat. Kalau nanti data live/backtest bertambah signifikan, ulangi
# check_concurrency.py untuk re-validasi, jangan anggap angka ini final
# selamanya.
SCALP_MAX_CONCURRENT_SAME_DIRECTION = 2

# State per-proses: {(sub_method_label, direction): {"time": Timestamp, "entry": float}}
# Modul-level SENGAJA (bukan argumen fungsi) supaya persist otomatis antar
# panggilan detect_scalp_cascade() berturut-turut -- baik loop live di
# server.py (tiap request /analyze-scalp, proses yang sama tetap hidup)
# maupun backtest candle-by-candle. Konsekuensinya: WAJIB panggil
# reset_scalp_cooldown_state() di awal tiap backtest run yang berdiri
# sendiri (lihat backtest_scalp_cascade.py) -- kalau tidak, run kedua di
# proses yang sama akan "mewarisi" cooldown residual dari run pertama.
_SCALP_LAST_SIGNAL = {}


def reset_scalp_cooldown_state():
    """Reset state cooldown/dedup sinyal scalp. WAJIB dipanggil di awal tiap
    backtest run (supaya run tidak mewarisi residual dari run sebelumnya di
    proses yang sama). Di live (server.py) TIDAK perlu dipanggil -- justru
    state yang persist antar request itu tujuannya."""
    _SCALP_LAST_SIGNAL.clear()


def _scalp_infer_bar_seconds(df: pd.DataFrame):
    """Infer lebar 1 bar (detik) dari 2 candle terakhir di df. Dibuat generik
    (bukan hardcode 300 detik/M5) supaya cooldown tetap benar kalau dipanggil
    dari timeframe lain (server.py mendukung M1-D1 utk /analyze-scalp)."""
    if "open_time" not in df.columns or len(df) < 2:
        return None
    delta = df["open_time"].iloc[-1] - df["open_time"].iloc[-2]
    secs = delta.total_seconds()
    return secs if secs > 0 else None


def _scalp_is_duplicate_signal(df: pd.DataFrame, sub_method_label: str,
                                direction: str, entry: float, sl: float) -> bool:
    """True kalau kandidat sinyal ini kemungkinan duplikat beruntun dari
    sinyal (sub_method + direction) sama yang baru saja fire -- jarak waktu
    (dikonversi ke 'jumlah bar', bukan hardcode) DAN (jarak entry ATAU SL
    identik), definisi sama persis dengan clustering di
    analyze_spread_and_clustering.py (termasuk fix SL 2026-09-26: SL
    identik = level struktural sama = tetap duplikat, walau entry bergeser
    jauh saat harga choppy di sekitar level itu).
    Fail-OPEN (return False) kalau tidak bisa dievaluasi (mis. tidak ada
    kolom open_time) -- supaya bug di sini gagal dengan cara yang KELIHATAN
    (sinyal tetap keluar apa adanya) bukan diam-diam memblokir semua sinyal."""
    key = (sub_method_label, direction)
    last = _SCALP_LAST_SIGNAL.get(key)
    if last is None or last.get("time") is None:
        return False
    bar_seconds = _scalp_infer_bar_seconds(df)
    if not bar_seconds:
        return False
    cur_time = df["open_time"].iloc[-1]
    bar_gap = (cur_time - last["time"]).total_seconds() / bar_seconds
    pip_gap = abs(entry - last["entry"]) / SCALP_PIP_SIZE
    sl_gap = abs(sl - last["sl"]) / SCALP_PIP_SIZE
    same_level = (pip_gap <= SCALP_COOLDOWN_PIP_GAP_MAX
                  or sl_gap <= SCALP_COOLDOWN_SL_GAP_MAX_PIP)
    return bar_gap <= SCALP_COOLDOWN_BAR_GAP_MAX and same_level


def _scalp_record_signal(df: pd.DataFrame, sub_method_label: str,
                          direction: str, entry: float, sl: float) -> None:
    cur_time = df["open_time"].iloc[-1] if "open_time" in df.columns else None
    _SCALP_LAST_SIGNAL[(sub_method_label, direction)] = {
        "time": cur_time, "entry": entry, "sl": sl,
    }


def _scalp_finalize(sub_method: int, label: str, direction: str, entry, sl,
                     entry_type: str, basis_note: str):
    """Bangun output setup scalping final + guard rentang SL wajar (pips)."""
    if direction not in ("bullish", "bearish") or entry is None or sl is None:
        return None
    entry = float(entry)
    sl = float(sl)
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    risk_pips = risk / SCALP_PIP_SIZE
    if risk_pips < SCALP_SL_MIN_PIPS or risk_pips > SCALP_SL_MAX_PIPS:
        return None
    rr_dir = 1 if direction == "bullish" else -1
    tp1 = entry + rr_dir * risk * SCALP_RR_TP[0]
    tp2 = entry + rr_dir * risk * SCALP_RR_TP[1]
    tp3 = entry + rr_dir * risk * SCALP_RR_TP[2]
    tier = classify_tier(risk, abs(tp1 - entry), abs(tp2 - entry))
    return {
        "sub_method": sub_method,
        "sub_method_label": label,
        "direction": "LONG" if direction == "bullish" else "SHORT",
        "entry_type": entry_type,
        "entry": round(entry, 4),
        "sl": round(sl, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "tp3": round(tp3, 4),
        "risk_pips": round(risk_pips, 1),
        "tier": tier,
        "basis_note": basis_note,
    }


def _scalp_has_displacement(df, start_idx, end_idx, direction,
                             body_mult=SCALP_DISPLACEMENT_BODY_MULT, hist_window=20):
    """Cari candle displacement (body besar, searah `direction`) di [start_idx, end_idx]."""
    bodies = (df["close"] - df["open"]).abs()
    hist_start = max(0, start_idx - hist_window)
    avg_body = bodies.iloc[hist_start:start_idx].mean()
    if not avg_body or avg_body <= 0 or np.isnan(avg_body):
        return None
    for i in range(start_idx, end_idx + 1):
        is_up = df["close"].iloc[i] > df["open"].iloc[i]
        same_dir = (direction == "bullish" and is_up) or (direction == "bearish" and not is_up)
        if same_dir and bodies.iloc[i] >= body_mult * avg_body:
            return i
    return None


def detect_pin_bar(df, idx, wick_body_ratio=SCALP_PIN_WICK_BODY_RATIO, close_pos_pct=0.6):
    """Pin bar / rejection candle sederhana di candle `idx`. Return 'bullish'/'bearish'/None."""
    o = float(df["open"].iloc[idx]); h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx]); c = float(df["close"].iloc[idx])
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o) or rng * 0.001
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if lower_wick >= wick_body_ratio * body and lower_wick > upper_wick * 1.2 and (c - l) >= rng * close_pos_pct:
        return "bullish"
    if upper_wick >= wick_body_ratio * body and upper_wick > lower_wick * 1.2 and (h - c) >= rng * close_pos_pct:
        return "bearish"
    return None


def _scalp_sub1_sweep_displacement_choch(df, pools, structure, recent_start, last_idx):
    """Sub 1: Sweep -> Displacement -> CHoCH/MSS searah -> entry LIMIT di edge FVG/OB."""
    sweeps = []
    for p in pools:
        seg_start = max(p["last_idx"] + 1, recent_start)
        for i in range(seg_start, last_idx + 1):
            if p["side"] == "buyside" and df["high"].iloc[i] > p["price"] and df["close"].iloc[i] < p["price"]:
                sweeps.append({"pool": p, "sweep_idx": i, "direction": "bearish"})
            elif p["side"] == "sellside" and df["low"].iloc[i] < p["price"] and df["close"].iloc[i] > p["price"]:
                sweeps.append({"pool": p, "sweep_idx": i, "direction": "bullish"})
    if not sweeps:
        return None
    sweeps.sort(key=lambda x: x["sweep_idx"], reverse=True)
    sw = sweeps[0]
    direction, sweep_idx = sw["direction"], sw["sweep_idx"]

    disp_idx = _scalp_has_displacement(df, sweep_idx, last_idx, direction)
    if disp_idx is None:
        return None

    bos, bos_idx = structure.get("bos"), structure.get("bos_idx")
    if not (bos is not None and bos_idx is not None and bos[0] == direction and bos_idx >= sweep_idx):
        return None  # CHoCH/MSS searah belum terkonfirmasi

    zone = None
    obs = mark_mitigated_ob(df, find_order_blocks(df, bos, bos_idx))
    for ob in obs:
        if ob["type"] == direction and not ob.get("mitigated") and ob["idx"] >= sweep_idx - 5:
            zone = {"low": ob["low"], "high": ob["high"], "kind": "OB"}
            break
    if zone is None:
        for g in find_fair_value_gaps(df):
            if g["type"] == direction and g["idx"] >= sweep_idx:
                zone = {"low": g["zone_low"], "high": g["zone_high"], "kind": "FVG"}
                break
    if zone is None:
        return None

    if direction == "bullish":
        entry = zone["high"]
        sl = min(float(df["low"].iloc[sweep_idx]), zone["low"]) - SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
    else:
        entry = zone["low"]
        sl = max(float(df["high"].iloc[sweep_idx]), zone["high"]) + SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE

    basis = (f"Sweep {sw['pool']['side']} @ {sw['pool']['price']:.2f} (candle #{sweep_idx}) -> "
             f"displacement (#{disp_idx}) -> CHoCH searah (BOS #{bos_idx}) -> "
             f"entry limit di edge {zone['kind']} {zone['low']:.2f}-{zone['high']:.2f}")
    return _scalp_finalize(1, "Sweep->Displacement->CHoCH->FVG/OB", direction, entry, sl, "limit", basis)


def _scalp_sub2_ob_retest(df, structure, recent_start, last_idx):
    """Sub 2: retest OB/S&D TANPA syarat sweep baru -> entry MARKET (retest sudah closed)."""
    bos, bos_idx = structure.get("bos"), structure.get("bos_idx")
    if bos is None:
        return None
    direction = bos[0]
    obs = mark_mitigated_ob(df, find_order_blocks(df, bos, bos_idx, max_lookback=60))
    for ob in obs:
        if ob["type"] != direction:
            continue
        for i in range(recent_start, last_idx + 1):
            if i <= ob["idx"]:
                continue
            touched = df["low"].iloc[i] <= ob["high"] and df["high"].iloc[i] >= ob["low"]
            if not touched:
                continue
            if direction == "bullish" and df["close"].iloc[i] > ob["high"]:
                entry = float(df["close"].iloc[i])
                sl = ob["low"] - SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
                basis = f"Retest OB bullish {ob['low']:.2f}-{ob['high']:.2f} @ candle #{i}, tanpa syarat sweep baru"
                return _scalp_finalize(2, "OB/S&D retest tanpa sweep", direction, entry, sl, "market", basis)
            if direction == "bearish" and df["close"].iloc[i] < ob["low"]:
                entry = float(df["close"].iloc[i])
                sl = ob["high"] + SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
                basis = f"Retest OB bearish {ob['low']:.2f}-{ob['high']:.2f} @ candle #{i}, tanpa syarat sweep baru"
                return _scalp_finalize(2, "OB/S&D retest tanpa sweep", direction, entry, sl, "market", basis)
    return None


def _scalp_sub3_bos_pullback(df, structure, recent_start, last_idx):
    """Sub 3: BOS continuation + pullback ke fib 50-61.8% -> entry LIMIT."""
    bos, bos_idx = structure.get("bos"), structure.get("bos_idx")
    if bos is None:
        return None
    direction, break_level = bos
    seg = df.iloc[bos_idx:last_idx + 1]
    if seg.empty:
        return None

    if direction == "bullish":
        impulse_end = float(seg["high"].max())
        if impulse_end <= break_level:
            return None
        rng = impulse_end - break_level
        fib_low, fib_high = impulse_end - rng * 0.618, impulse_end - rng * 0.5
        cur_low = float(df["low"].iloc[last_idx])
        if not (fib_low <= cur_low <= fib_high):
            return None
        entry = fib_high
        sl = break_level - SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
    else:
        impulse_end = float(seg["low"].min())
        if impulse_end >= break_level:
            return None
        rng = break_level - impulse_end
        fib_high, fib_low = impulse_end + rng * 0.618, impulse_end + rng * 0.5
        cur_high = float(df["high"].iloc[last_idx])
        if not (fib_low <= cur_high <= fib_high):
            return None
        entry = fib_low
        sl = break_level + SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE

    basis = (f"BOS {direction} @ {break_level:.2f} (candle #{bos_idx}), harga sekarang di fib "
             f"pullback 50-61.8% dari impulse leg")
    return _scalp_finalize(3, "BOS continuation + pullback", direction, entry, sl, "limit", basis)


def _scalp_sub4_pin_bar_key_level(df, pools, recent_start, last_idx, tol_pct=SCALP_KEY_LEVEL_TOL_PCT):
    """Sub 4: rejection candle/pin bar di key level (liquidity pool) -> entry MARKET."""
    key_levels = [p["price"] for p in pools]
    if not key_levels:
        return None
    for i in range(last_idx, recent_start - 1, -1):
        pin = detect_pin_bar(df, i)
        if pin is None:
            continue
        px_low, px_high = float(df["low"].iloc[i]), float(df["high"].iloc[i])
        ref = px_low if pin == "bullish" else px_high
        if not any(abs(ref - lvl) / lvl <= tol_pct for lvl in key_levels):
            continue
        entry = float(df["close"].iloc[i])
        if pin == "bullish":
            sl = px_low - SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
        else:
            sl = px_high + SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
        basis = f"Pin bar {pin} @ candle #{i} dekat key level {ref:.2f}"
        return _scalp_finalize(4, "Rejection candle/pin bar di key level", pin, entry, sl, "market", basis)
    return None


def _scalp_sub5_failed_breakout(df, pools, swing_highs, swing_lows, recent_start, last_idx):
    """Sub 5: failed breakout/failed auction di key level -> entry MARKET."""
    levels = [("buyside", p["price"]) for p in pools if p["side"] == "buyside"]
    levels += [("sellside", p["price"]) for p in pools if p["side"] == "sellside"]
    levels += [("buyside", pr) for _, _, pr in swing_highs[-5:]]
    levels += [("sellside", pr) for _, _, pr in swing_lows[-5:]]
    for side, lvl in levels:
        for i in range(recent_start, last_idx + 1):
            if side == "buyside" and df["high"].iloc[i] > lvl:
                seg = df.iloc[i:last_idx + 1]
                if (seg["close"] < lvl).iloc[-1]:
                    entry = float(df["close"].iloc[last_idx])
                    sl = float(seg["high"].max()) + SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
                    basis = f"Failed breakout di atas {lvl:.2f} (candle #{i}), close kembali di bawah level"
                    return _scalp_finalize(5, "Failed breakout/failed auction", "bearish", entry, sl, "market", basis)
            if side == "sellside" and df["low"].iloc[i] < lvl:
                seg = df.iloc[i:last_idx + 1]
                if (seg["close"] > lvl).iloc[-1]:
                    entry = float(df["close"].iloc[last_idx])
                    sl = float(seg["low"].min()) - SCALP_SL_BUFFER_PIPS * SCALP_PIP_SIZE
                    basis = f"Failed breakout di bawah {lvl:.2f} (candle #{i}), close kembali di atas level"
                    return _scalp_finalize(5, "Failed breakout/failed auction", "bullish", entry, sl, "market", basis)
    return None


def detect_scalp_cascade(df: pd.DataFrame, lookback_candles: int = SCALP_LOOKBACK_CANDLES,
                          open_positions: dict | None = None) -> dict:
    """
    Dispatcher cascade waterfall: coba sub 1..5 berurutan, kembalikan match
    PERTAMA yang lolos guard SL. XAUUSD-only -- jangan panggil untuk crypto.
    df harus struktur asli XAUUSD (df_structure dari get_structure_df, sama
    seperti dipakai analyze_xau), BUKAN proxy PAXGUSDT kalau bisa dihindari.

    open_positions: dict opsional {"LONG": n, "SHORT": n} -- jumlah posisi
    SEARAH yang CALLER laporkan sendiri sedang terbuka di broker saat ini
    (lihat catatan SCALP_MAX_CONCURRENT_SAME_DIRECTION di atas). Default None
    -> diperlakukan {"LONG": 0, "SHORT": 0}, cap tidak pernah aktif (dipakai
    apa adanya oleh backtest_scalp_cascade.py, yang tidak mensimulasikan
    status posisi riil).
    """
    open_positions = open_positions or {}
    swing_highs, swing_lows = find_swings(df)
    structure = classify_structure(df, swing_highs, swing_lows)
    pools = mark_swept_pools(df, find_liquidity_pools(swing_highs, swing_lows))
    last_idx = len(df) - 1
    recent_start = max(0, last_idx - lookback_candles + 1)

    capped_candidates = []  # utk transparansi kalau semua kandidat kena cap

    for fn, args in [
        (_scalp_sub1_sweep_displacement_choch, (df, pools, structure, recent_start, last_idx)),
        (_scalp_sub2_ob_retest, (df, structure, recent_start, last_idx)),
        (_scalp_sub3_bos_pullback, (df, structure, recent_start, last_idx)),
        (_scalp_sub4_pin_bar_key_level, (df, pools, recent_start, last_idx)),
        (_scalp_sub5_failed_breakout, (df, pools, swing_highs, swing_lows, recent_start, last_idx)),
    ]:
        result = fn(*args)
        if not result:
            continue
        # Cooldown/dedup: kalau kandidat ini kemungkinan duplikat beruntun
        # dari sinyal (sub_method + direction) yang sama, JANGAN langsung
        # hentikan cascade -- coba sub-metode berikutnya di waterfall,
        # karena "sub 1 lagi cooldown" bukan berarti "tidak ada sinyal valid
        # sama sekali di candle ini".
        if _scalp_is_duplicate_signal(df, result["sub_method_label"],
                                       result["direction"], result["entry"],
                                       result["sl"]):
            continue
        # Concurrency cap SEARAH (lintas sub-metode) -- lihat catatan di
        # SCALP_MAX_CONCURRENT_SAME_DIRECTION. Kalau caller melaporkan sudah
        # ada >= cap posisi searah terbuka, kandidat ini di-skip (bukan
        # otomatis dihentikan seluruh cascade -- arah BERLAWANAN tetap boleh
        # lolos, karena itu bukan stacking risk yang sama).
        if open_positions.get(result["direction"], 0) >= SCALP_MAX_CONCURRENT_SAME_DIRECTION:
            capped_candidates.append(result)
            continue
        _scalp_record_signal(df, result["sub_method_label"],
                              result["direction"], result["entry"], result["sl"])
        return result

    if capped_candidates:
        blocked = capped_candidates[0]
        return {
            "sub_method": None,
            "sub_method_label": None,
            "direction": "NEUTRAL",
            "blocked_by_concurrency_cap": True,
            "reason": (
                f"Kandidat valid ada ({blocked['sub_method_label']}, "
                f"{blocked['direction']}) tapi ditahan: caller melaporkan "
                f"sudah {open_positions.get(blocked['direction'], 0)} posisi "
                f"{blocked['direction']} terbuka, >= cap "
                f"SCALP_MAX_CONCURRENT_SAME_DIRECTION="
                f"{SCALP_MAX_CONCURRENT_SAME_DIRECTION}. Tutup/kurangi posisi "
                f"searah dulu di broker sebelum entry baru arah yang sama."
            ),
        }

    return {
        "sub_method": None,
        "sub_method_label": None,
        "direction": "NEUTRAL",
        "reason": f"Tidak ada trigger valid di 5 sub-metode dalam {lookback_candles} candle terakhir "
                  f"(atau kandidat yang ada masih dalam cooldown duplikat sinyal).",
    }


# =============================================================================
# 10. MAIN
# =============================================================================
def run(symbol, interval, rr=2.0, chart=False, cot=False, dxy_bias=None,
        only_tradeable=False):
    display, bsym, is_alias = resolve_symbol(symbol)
    bint, tf_label = parse_timeframe(interval)
    asset_class = "xau" if is_alias else "crypto"

    df = fetch_klines(bsym, bint, limit=300)
    has_futures = symbol_has_futures(bsym)

    if asset_class == "crypto":
        a = analyze_crypto(df, bsym, has_futures)
    else:
        df_structure, structure_source = get_structure_df(asset_class, tf_label, df)
        a = analyze_xau(df, bsym, has_futures, dxy_bias_override=dxy_bias,
                         df_structure=df_structure, structure_source=structure_source)

    # "a['price']" dari analyze_crypto/xau = close candle H1/H4/dst yang SUDAH
    # closed (sengaja, biar struktur/VWAP/POC tidak goyang oleh candle yang
    # masih berjalan). Tapi itu tidak cocok dipakai sebagai "harga sekarang"
    # untuk Entry -> bisa basi sampai hampir 1 candle penuh. Override dengan
    # harga live (ticker real-time) khusus untuk Entry/SL/TP; struktur tetap
    # dari candle closed di atas.
    live_price = fetch_live_price(bsym)
    a["price_closed_candle"] = a["price"]
    if live_price:
        a["price"] = live_price
        a["price_is_live"] = True
    else:
        a["price_is_live"] = False

    s = build_setup(a, rr=rr)
    spot_price, spot_label = fetch_spot_crosscheck(bsym)
    cot_data = fetch_cot_gold() if (cot and asset_class == "xau") else None

    # Crypto tidak punya spot_price terpisah (SPOT_CHECK hanya utk PAXG) -> sebelumnya
    # type=None -> label generik "LIMIT/MARKET" -> tradeable TIDAK PERNAH True utk
    # BTC/ETH di jalur live (padahal backtest memakai harga bar sbg referensi).
    # Fix: utk crypto, pakai harga live Binance sbg referensi (sumber yg sama dgn entry).
    ref_price = spot_price if spot_price else (a["price"] if asset_class == "crypto" else None)
    order_type_detail = classify_order_type(s["direction"], s["entry"], ref_price)
    resolved_order_type = order_type_detail["type"] or s["order_type"]
    tradeable_eval = evaluate_tradeable(a, s, order_type=resolved_order_type,
                                       asset_class=asset_class, tf_label=tf_label)

    # Mode opsional --only-tradeable: hanya tampilkan setup kalau Tier A DAN
    # tradeable=True. Kalau tidak, JANGAN cetak entry/SL/TP (supaya tidak
    # disalahgunakan), cukup alasan kenapa tidak lolos. Default MATI --
    # SOP standar tetap menampilkan semua tier apa adanya.
    if only_tradeable and not tradeable_eval["tradeable"]:
        print(f"\n[{display} {tf_label}] TIDAK ADA SETUP yang memenuhi syarat "
              f"(wajib Tier A + tradeable=True) saat ini.")
        print(f"  Tier saat ini : {s.get('tier')}")
        print(f"  Alasan        : {tradeable_eval['reason']}")
        print("  Coba lagi saat candle berikutnya close.\n")
        return a, s

    print_report(display, bsym, tf_label, a, s, asset_class,
                 is_alias=is_alias, spot_price=spot_price, spot_label=spot_label, cot=cot_data)
    tag = "LAYAK (OB asli + LIMIT pending, tidak mengejar harga)" if tradeable_eval["tradeable"] else "MARJINAL/SKIP"
    print(f"  [FILTER KONSERVATIF] {tag} -- {tradeable_eval['reason']}\n")

    if chart:
        try:
            plot_chart(display, bsym, tf_label, df, a, s)
        except Exception as e:
            print(f"  [chart gagal: {e}]")
    return a, s


def main():
    ap = argparse.ArgumentParser(description="AI Trading Analysis v2 (Liquidity/Structure-based)")
    ap.add_argument("symbol", help="Simbol: BTCUSDT/ETHUSDT, atau XAUUSD/XAU/GOLD (emas)")
    ap.add_argument("interval", help="Timeframe: M1, M5, M15, M30, H1, H4, D1")
    ap.add_argument("--rr", type=float, default=2.0, help="Target Risk:Reward TP2 (default 2)")
    ap.add_argument("--chart", action="store_true", help="Simpan chart PNG")
    ap.add_argument("--cot", action="store_true", help="Tampilkan COT Gold (informasional, khusus XAU)")
    ap.add_argument("--dxy-bias", choices=["bullish", "bearish", "neutral"], default=None,
                     help="Override manual bias DXY/macro (khusus XAU), karena real yield & "
                          "ekspektasi Fed tidak tersedia via API gratis")
    ap.add_argument("--only-tradeable", action="store_true",
                     help="Hanya tampilkan setup kalau Tier A DAN tradeable=True; kalau tidak, "
                          "cuma cetak alasannya (tanpa entry/SL/TP)")
    ap.add_argument("--list", nargs="+", metavar="SYM", help="Analisis banyak simbol sekaligus")
    args = ap.parse_args()

    targets = args.list if args.list else [args.symbol]
    for sym in targets:
        try:
            run(sym, args.interval, rr=args.rr, chart=args.chart, cot=args.cot,
                dxy_bias=args.dxy_bias, only_tradeable=args.only_tradeable)
        except ValueError as e:
            print(f"[error] {e}")
            if not args.list:
                sys.exit(2)
        except Exception as e:
            print(f"[{sym}] gagal: {e}")


if __name__ == "__main__":
    main()
