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
    result = {"bias": "ranging", "detail": "", "bos": None}
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
    if swing_highs:
        last_swing_high = swing_highs[-1][2]
        if last_close > last_swing_high:
            result["bos"] = ("bullish", last_swing_high)
    if swing_lows:
        last_swing_low = swing_lows[-1][2]
        if last_close < last_swing_low:
            # kalau dua-duanya break (jarang), yang paling baru menang -> cek index
            if result["bos"] is None or swing_lows[-1][0] > swing_highs[-1][0]:
                result["bos"] = ("bearish", last_swing_low)
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
    return {"poc": poc_price, "vah": vah, "val": val, "edges": edges, "vol_bins": vol_bins}


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

    return dict(
        score=score, reasons=reasons, price=price,
        structure=structure, pools=pools, fvgs=fvgs, vp=vp, vwap=vwap_val,
        delta=delta_now, cvd_trend=cvd_trend, funding=funding, oi=oi, magnet=magnet,
        swing_h=swing_h, swing_l=swing_l,
    )


# =============================================================================
# 6. SCORING ENGINE — XAU (via PAXGUSDT, playbook hibrida)
# =============================================================================
def analyze_xau(df: pd.DataFrame, symbol: str, has_futures: bool, dxy_bias_override=None) -> dict:
    price = float(df["close"].iloc[-1])
    swing_h, swing_l = find_swings(df, left=3, right=3)
    structure = classify_structure(df, swing_h, swing_l)
    pools = find_liquidity_pools(swing_h, swing_l)
    pools = mark_swept_pools(df, pools)
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

    return dict(
        score=score, reasons=reasons, price=price,
        structure=structure, pools=pools, vp=vp, vwap=vwap_val,
        reaction=reaction, vol_ratio=vol_ratio, dxy=dxy_info,
        dxy_override=dxy_bias_override, funding=funding, oi=oi, magnet=magnet,
        swing_h=swing_h, swing_l=swing_l, session_start=session_start,
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

    def targets_above(val, tol_ratio=0.0015):
        """Semua level pool/swing di atas val, urut terdekat->terjauh,
        level yang berhimpitan (<tol_ratio dari val) digabung jadi satu."""
        cands = [p["price"] for p in pools if p["price"] > val]
        if swing_h:
            cands += [s[2] for s in swing_h if s[2] > val]
        cands = sorted(set(cands))
        merged = []
        for c in cands:
            if not merged or (c - merged[-1]) > val * tol_ratio:
                merged.append(c)
        return merged

    def targets_below(val, tol_ratio=0.0015):
        """Sama seperti targets_above tapi ke bawah, urut terdekat->terjauh."""
        cands = [p["price"] for p in pools if p["price"] < val]
        if swing_l:
            cands += [s[2] for s in swing_l if s[2] < val]
        cands = sorted(set(cands), reverse=True)
        merged = []
        for c in cands:
            if not merged or (merged[-1] - c) > val * tol_ratio:
                merged.append(c)
        return merged

    # Jarak SL akhir (dari entry) TIDAK BOLEH lebih kecil dari ini, apa pun
    # posisi liquidity pool/swing terdekat. Sebelumnya floor ini cuma dipakai
    # untuk menghitung besar buffer, bukan untuk menjamin jarak SL akhir -->
    # itu penyebab SL bisa jadi cuma beberapa cent dari harga kalau struktur
    # kebetulan sangat dekat. Sekarang floor ini yang menentukan jarak minimum.
    MIN_SL_PCT = 0.001  # 0.1% dari harga; naikkan kalau masih terasa terlalu ketat

    if direction == "LONG":
        sl_level = nearest_below(price)
        if sl_level is None:
            sl_level = price * 0.985
        raw_dist = price - sl_level
        min_dist = price * MIN_SL_PCT
        dist = max(raw_dist, min_dist)
        buffer_ = dist * 0.10
        entry = price
        sl = entry - dist - buffer_
        sl_dist_final = entry - sl

        # TP1/TP2/TP3 = pool/swing ke-1/2/3 terdekat searah trade (LOGIC
        # LIQUIDITY-BASED, konsisten dgn filosofi engine ini). Kalau struktur
        # yang tersedia kurang dari 3 level, atau levelnya kurang jauh (belum
        # 30% dari jarak SL -> terlalu dekat buat jadi target berarti), baru
        # fallback ke kelipatan R (1R/2R/3R) seperti versi lama.
        cand_targets = targets_above(price)
        min_gap = sl_dist_final * 0.3

        tp1 = (cand_targets[0] if len(cand_targets) >= 1
               and cand_targets[0] - entry >= min_gap
               else entry + sl_dist_final * 1.0)
        # Fallback TP2/TP3 dihitung relatif terhadap R yang sudah dicapai TP1,
        # bukan rr tetap -> supaya kalau TP1 (struktur asli) sudah jauh
        # (misal 3R), TP2 tidak jatuh SEBELUM TP1 (yang bikin urutan kebalik).
        r1_reached = (tp1 - entry) / sl_dist_final if sl_dist_final > 0 else 1.0
        tp2 = (cand_targets[1] if len(cand_targets) >= 2
               and cand_targets[1] > tp1
               else entry + sl_dist_final * max(rr, r1_reached + 1.0))
        r2_reached = (tp2 - entry) / sl_dist_final if sl_dist_final > 0 else 1.0
        tp3 = (cand_targets[2] if len(cand_targets) >= 3
               and cand_targets[2] > tp2
               else entry + sl_dist_final * max(rr * 1.5, r2_reached + 1.0))
        # Jaga-jaga terakhir supaya urutan tetap naik (TP1 < TP2 < TP3).
        tp2 = max(tp2, tp1 + sl_dist_final * 0.1)
        tp3 = max(tp3, tp2 + sl_dist_final * 0.1)

        risk = entry - sl
        order_type = "LIMIT/MARKET (struktur bullish)"
    elif direction == "SHORT":
        sl_level = nearest_above(price)
        if sl_level is None:
            sl_level = price * 1.015
        raw_dist = sl_level - price
        min_dist = price * MIN_SL_PCT
        dist = max(raw_dist, min_dist)
        buffer_ = dist * 0.10
        entry = price
        sl = entry + dist + buffer_
        sl_dist_final = sl - entry

        cand_targets = targets_below(price)
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
    else:
        entry = price
        sl = tp1 = tp2 = tp3 = price
        risk = 0.0
        order_type = None

    return dict(direction=direction, entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                rr=rr, risk=risk, conf=conf, order_type=order_type, price_now=price,
                poc=vp["poc"], vah=vp["vah"], val=vp["val"])


# =============================================================================
# 8. REPORT PRINTER
# =============================================================================
def fmt(x, dp=4):
    return f"{x:.{dp}f}"


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
        print(f"  ORDER TYPE  : {s['order_type']}")
        print(f"  ENTRY       : {fmt(s['entry'])}")
        print(f"  STOP LOSS   : {fmt(s['sl'])}   (di luar liquidity pool/swing terdekat + buffer)")
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

    print("  STRUKTUR & LIKUIDITAS")
    print(f"   Bias HTF     : {a['structure']['bias'].upper()} — {a['structure']['detail']}")
    if a["structure"]["bos"]:
        side, lvl = a["structure"]["bos"]
        print(f"   BOS terakhir : {side} menembus {fmt(lvl)}")
    print(f"   POC / VAH / VAL : {fmt(a['vp']['poc'])} / {fmt(a['vp']['vah'])} / {fmt(a['vp']['val'])}")
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
def plot_chart(display_symbol, binance_symbol, tf_label, df, a, s, window=120):
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
    plt.savefig(fname, dpi=110)
    plt.close()
    print(f"  [chart disimpan: {fname}]")


# =============================================================================
# 10. MAIN
# =============================================================================
def run(symbol, interval, rr=2.0, chart=False, cot=False, dxy_bias=None):
    display, bsym, is_alias = resolve_symbol(symbol)
    bint, tf_label = parse_timeframe(interval)
    asset_class = "xau" if is_alias else "crypto"

    df = fetch_klines(bsym, bint, limit=300)
    has_futures = symbol_has_futures(bsym)

    if asset_class == "crypto":
        a = analyze_crypto(df, bsym, has_futures)
    else:
        a = analyze_xau(df, bsym, has_futures, dxy_bias_override=dxy_bias)

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

    print_report(display, bsym, tf_label, a, s, asset_class,
                 is_alias=is_alias, spot_price=spot_price, spot_label=spot_label, cot=cot_data)

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
    ap.add_argument("--list", nargs="+", metavar="SYM", help="Analisis banyak simbol sekaligus")
    args = ap.parse_args()

    targets = args.list if args.list else [args.symbol]
    for sym in targets:
        try:
            run(sym, args.interval, rr=args.rr, chart=args.chart, cot=args.cot, dxy_bias=args.dxy_bias)
        except ValueError as e:
            print(f"[error] {e}")
            if not args.list:
                sys.exit(2)
        except Exception as e:
            print(f"[{sym}] gagal: {e}")


if __name__ == "__main__":
    main()
