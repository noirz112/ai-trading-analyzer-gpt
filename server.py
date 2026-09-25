#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — Web API (FastAPI) wrapper untuk engine.py
=============================================================================
engine.py adalah CLI script (argparse, wajib symbol+interval). Kalau
di-deploy ke Railway sebagai Service tanpa wrapper ini, proses langsung
exit(2) begitu start tanpa argumen -> Railway tandai CRASHED -> restart ->
crash lagi. Itu penyebab error 502 di Custom GPT.

server.py ini bikin engine.py "hidup" sebagai web server (listen di $PORT),
dipanggil Custom GPT lewat HTTP GET /analyze, bukan command line.

ENDPOINT:
  GET /health
  GET /analyze?symbol=XAUUSD&interval=H1&rr=2&chart=false&cot=false&dxy_bias=neutral
  GET /analyze-scalp?symbol=XAUUSD&interval=M5   <- KHUSUS XAUUSD, cascade waterfall
                                                     5 sub-metode (lihat detect_scalp_cascade
                                                     di engine.py). Parameter internal BELUM
                                                     dikalibrasi lewat backtest -- pakai dulu
                                                     utk forward-test akun DEMO + backtest data
                                                     historis paralel, JANGAN live dgn uang riil
                                                     sebelum kalibrasi selesai.
  GET /openapi.json   <- otomatis dari FastAPI, tinggal di-import ke GPT Action

DEPLOY DI RAILWAY:
  1. requirements.txt: fastapi, uvicorn[standard], pandas, numpy, requests,
     pydantic, matplotlib (matplotlib wajib ada kalau mau pakai --chart)
  2. Environment variable: TWELVEDATA_API_KEY (untuk struktur XAUUSD asli --
     SOP hybrid, lihat get_structure_df() di engine.py). Kalau belum di-set,
     server TETAP JALAN, otomatis fallback ke struktur PAXGUSDT lama (ditandai
     "structure_source": "paxg_fallback_no_api_key" di response) -- bukan crash.
  3. Start command:
       uvicorn server:app --host 0.0.0.0 --port $PORT
     (JANGAN "python engine.py ..." lagi — itu penyebab crash)
"""
import io
import os
import contextlib
import traceback

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import (
    resolve_symbol, parse_timeframe, fetch_klines, symbol_has_futures,
    analyze_crypto, analyze_xau, build_setup, fetch_spot_crosscheck,
    fetch_cot_gold, fetch_live_price, print_report, plot_chart,
    classify_order_type, get_structure_df, evaluate_tradeable,
    detect_scalp_cascade, SCALP_LOOKBACK_CANDLES, SCALP_SL_MIN_PIPS,
    SCALP_SL_MAX_PIPS, SCALP_PIP_SIZE, SCALP_MAX_CONCURRENT_SAME_DIRECTION,
)

app = FastAPI(
    title="AI Trading Analyzer API",
    description="Backend untuk Custom GPT — analisa liquidity/structure-based XAU & crypto.",
    version="1.0.0",
    servers=[
        {"url": "https://ai-trading-analyzer-gpt-production.up.railway.app"},
    ],
)

# Default mode server. Set env ONLY_TRADEABLE_DEFAULT=true di Railway supaya SEMUA
# panggilan /analyze (termasuk dari Custom GPT yang belum tahu parameter
# only_tradeable) hanya mengembalikan setup kalau Tier A + tradeable=True.
# Tidak di-set / false = perilaku lama (semua tier tampil).
ONLY_TRADEABLE_DEFAULT = os.environ.get("ONLY_TRADEABLE_DEFAULT", "false").strip().lower() in ("1", "true", "yes", "on")

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHART_DIR, exist_ok=True)
app.mount("/charts", StaticFiles(directory=CHART_DIR), name="charts")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="BTCUSDT/ETHUSDT, atau XAUUSD/XAU/GOLD"),
    interval: str = Query(..., description="M1, M5, M15, M30, H1, H4, D1"),
    rr: float = Query(2.0, description="Target Risk:Reward TP2"),
    chart: bool = Query(False, description="Simpan & kembalikan chart PNG"),
    cot: bool = Query(False, description="Sertakan COT Gold (khusus XAU)"),
    dxy_bias: str = Query(None, description="bullish/bearish/neutral (khusus XAU, opsional)"),
    only_tradeable: bool = Query(None, description="Kalau true: hanya kembalikan setup jika "
                                 "Tier A DAN tradeable=True; kalau tidak, kembalikan alasan tanpa entry/SL/TP. "
                                 "Kosong = ikut default server (env ONLY_TRADEABLE_DEFAULT)."),
):
    if only_tradeable is None:
        only_tradeable = ONLY_TRADEABLE_DEFAULT

    if not symbol or not interval:
        return JSONResponse(
            status_code=400,
            content={"error": "Parameter 'symbol' dan 'interval' wajib diisi."},
        )

    try:
        display, bsym, is_alias = resolve_symbol(symbol)
        bint, tf_label = parse_timeframe(interval)
        asset_class = "xau" if is_alias else "crypto"

        df = fetch_klines(bsym, bint, limit=300)
        has_futures = symbol_has_futures(bsym)

        if asset_class == "crypto":
            a = analyze_crypto(df, bsym, has_futures)
        else:
            # SOP hybrid: struktur (swing/BOS/liquidity pool/Order Block) dari
            # TwelveData (XAUUSD asli) kalau tersedia, fallback ke PAXGUSDT
            # kalau API key belum ada / request gagal -- lihat get_structure_df().
            df_structure, structure_source = get_structure_df(asset_class, tf_label, df)
            a = analyze_xau(df, bsym, has_futures, dxy_bias_override=dxy_bias,
                             df_structure=df_structure, structure_source=structure_source)

        # PENTING: server.py punya jalur eksekusi SENDIRI, terpisah dari
        # run()/main() di engine.py (yang cuma dipakai CLI). Override harga
        # live harus dipasang DI SINI juga, bukan cuma di run() -- kalau
        # tidak, endpoint /analyze yang dipakai GPT tidak pernah dapat
        # manfaatnya sama sekali (ini penyebab bug 'live price gagal terus').
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

        # PENTING: entry/sl/tp di atas dihitung dari harga proxy (PAXGUSDT
        # atau gold-api), BUKAN dari feed broker user (mis. icMarkets).
        # Selisih antara keduanya bisa $1-5 tergantung venue/lag. Kalau ini
        # tidak ditonjolkan sebagai field terpisah, GPT sering skip
        # menyebutkannya dan user menyangka entry itu langsung market-ready
        # di broker mereka. order_type juga diklasifikasikan ulang di sini
        # (BUY/SELL LIMIT vs STOP vs MARKET) dengan membandingkan entry ke
        # spot_price -- bukan sekadar label generik "LIMIT/MARKET".
        # Crypto: tidak ada spot_price terpisah -> pakai harga live Binance sbg referensi
        # (fix: tanpa ini tradeable tidak pernah True utk BTC/ETH, tidak konsisten dgn backtest).
        ref_price = spot_price if spot_price else (a["price"] if asset_class == "crypto" else None)
        order_type_detail = classify_order_type(s["direction"], s["entry"], ref_price)
        resolved_order_type = order_type_detail["type"] or s["order_type"]

        # Filter konservatif tambahan (lihat evaluate_tradeable() di engine.py):
        # tradeable=True lewat jalur "confirmed" (bias mapan) ATAU jalur
        # "fresh_ob_pending" (OB asli + order LIMIT, tidak mengejar harga).
        # TIDAK menyembunyikan setup Tier B/C -- laporan tetap tampil penuh.
        tradeable_eval = evaluate_tradeable(a, s, order_type=resolved_order_type,
                                           asset_class=asset_class, tf_label=tf_label)
        if only_tradeable and not tradeable_eval["tradeable"]:
            return {
                "symbol": display,
                "interval": tf_label,
                "tradeable": False,
                "tier": s.get("tier"),
                "setup_available": False,
                "tradeable_reason": tradeable_eval["reason"],
                "message": "Tidak ada setup yang memenuhi syarat (wajib Tier A + tradeable=True) "
                           "saat ini. Entry/SL/TP sengaja tidak dikembalikan.",
                "instruction_for_assistant": "Sampaikan bahwa belum ada setup yang memenuhi syarat beserta "
                                             "tradeable_reason. JANGAN membuat atau menebak entry/SL/TP sendiri.",
            }
        spot_spread = None
        spot_spread_pct = None
        if spot_price:
            spot_spread = a["price"] - spot_price
            spot_spread_pct = (spot_spread / spot_price * 100) if spot_price else None

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(display, bsym, tf_label, a, s, asset_class,
                         is_alias=is_alias, spot_price=spot_price,
                         spot_label=spot_label, cot=cot_data)
        report_text = buf.getvalue()

        chart_url = None
        if chart:
            try:
                # TIDAK pakai os.chdir(): itu mengubah cwd milik SELURUH
                # proses (bukan per-thread), jadi request /analyze?chart=true
                # yang datang bersamaan (FastAPI menjalankan endpoint sync di
                # threadpool) bisa saling menimpa cwd satu sama lain dan
                # menyimpan PNG di direktori yang salah -> chart_url 404.
                # plot_chart() sekarang terima out_dir eksplisit dan
                # menyimpan langsung ke situ, aman untuk concurrency.
                fpath = plot_chart(display, bsym, tf_label, df, a, s, out_dir=CHART_DIR)
                fname = os.path.basename(fpath)
                chart_url = f"/charts/{fname}"
            except Exception as e:
                report_text += f"\n[chart gagal: {e}]"

        ob = s.get("order_block")
        entry_basis_note = {
            "edge_ob": "entry = edge Order Block (lihat field 'order_block'); "
                       "bandingkan tetap ke spot_price/spot_spread sebelum eksekusi",
            "swing_fallback_no_ob": "FALLBACK: tidak ada Order Block valid searah trade, "
                                     "entry = harga sekarang, SL dari nearest swing/pool + buffer",
        }.get(s.get("entry_basis"), "harga proxy (PAXGUSDT/gold-api), bukan feed broker -- "
                                     "lihat spot_price/spot_spread")

        return {
            "symbol": display,
            "interval": tf_label,
            "asset_class": asset_class,
            "structure_source": a.get("structure_source", "native"),
            "direction": s["direction"],
            "entry": s["entry"],
            "entry_basis": entry_basis_note,
            "order_block": ({
                "type": ob["type"], "low": ob["low"], "high": ob["high"],
            } if ob else None),
            "tier": s.get("tier"),
            "tradeable": tradeable_eval["tradeable"],
            "tradeable_path": tradeable_eval["path"],
            "tradeable_reason": tradeable_eval["reason"],
            "sl": s["sl"],
            "tp1": s["tp1"],
            "tp2": s["tp2"],
            "tp3": s["tp3"],
            "confidence": round(s["conf"], 1),
            "score": a["score"],
            "price_is_live": a.get("price_is_live", False),
            # Harga referensi spot XAUUSD real (jauh lebih dekat ke broker
            # user daripada proxy PAXGUSDT), dan selisihnya terhadap harga
            # yang dipakai untuk hitung entry.
            "spot_price": spot_price,
            "spot_label": spot_label,
            "spot_spread": spot_spread,
            "spot_spread_pct": spot_spread_pct,
            # Klasifikasi order berdasarkan posisi entry vs spot_price:
            # BUY/SELL LIMIT atau STOP = pending order (harga belum sampai
            # level entry), MARKET = entry sudah dekat harga sekarang.
            # type bisa None kalau spot_price gagal diambil.
            "order_type": order_type_detail["type"] or s["order_type"],
            "order_type_note": order_type_detail["note"],
            "report": report_text,
            "chart_url": chart_url,
        }

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        # Selalu balas JSON, jangan sampai exception bikin proses/response mati
        # -> GPT dapat pesan error jelas, bukan 502.
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Analisa gagal: {e}"})


@app.get("/analyze-scalp")
def analyze_scalp(
    symbol: str = Query("XAUUSD", description="KHUSUS XAUUSD/XAU/GOLD -- endpoint ini menolak simbol crypto."),
    interval: str = Query(..., description="M1, M5, M15, M30, H1, H4, D1 (scalp SOP: idealnya M1-M15)"),
    open_long: int = Query(0, ge=0, description="WAJIB diisi jujur oleh caller (user/Custom GPT): "
                            "berapa posisi LONG dari sistem ini yang SAAT INI masih terbuka di MT5. "
                            "Server TIDAK bisa tahu ini sendiri (tidak ada koneksi broker) -- kalau "
                            "diisi 0 padahal sebenarnya ada posisi terbuka, concurrency cap di bawah "
                            "TIDAK akan bekerja."),
    open_short: int = Query(0, ge=0, description="Sama seperti open_long, tapi untuk posisi SHORT."),
):
    """
    Cascade waterfall 5 sub-metode scalping (lihat detect_scalp_cascade() di
    engine.py) -- HANYA untuk XAUUSD, tidak menyentuh logika crypto sama
    sekali. Beda total dari /analyze (mode swing): dispatcher coba sub 1
    (sweep->displacement->CHoCH->FVG/OB) sampai sub 5 (failed breakout)
    berurutan, kembalikan match PERTAMA yang lolos guard SL 8-40 pips
    (SCALP_SL_MIN_PIPS/MAX_PIPS), hanya melihat 5 candle terakhir
    (SCALP_LOOKBACK_CANDLES) untuk trigger.

    PENTING -- parameter displacement body-mult, rasio wick/body pin bar,
    toleransi key level, dan rentang SL 8-40 pips di engine.py masih nilai
    desain awal, BELUM divalidasi lewat backtest per sub-metode. Endpoint ini
    dipakai untuk MENGUMPULKAN DATA (forward-test akun demo + backtest data
    historis secara paralel), bukan sinyal siap-eksekusi dengan uang riil.

    CONCURRENCY CAP (2026-09-26): eksekusi sistem ini MANUAL (user baca
    sinyal, buka posisi sendiri di MT5) -- server ini TIDAK PERNAH tahu
    posisi mana yang masih terbuka di broker. Maka cap posisi searah
    (SCALP_MAX_CONCURRENT_SAME_DIRECTION di engine.py) hanya bisa ditegakkan
    kalau open_long/open_short di atas dilaporkan JUJUR oleh caller di setiap
    panggilan -- ini BUKAN pengaman otomatis, ini pengaman yang bergantung
    pada laporan manual. Kalau parameter ini tidak diisi (default 0), cap
    efeknya tidak aktif sama sekali.
    """
    if not symbol or not interval:
        return JSONResponse(
            status_code=400,
            content={"error": "Parameter 'symbol' dan 'interval' wajib diisi."},
        )

    try:
        display, bsym, is_alias = resolve_symbol(symbol)
        bint, tf_label = parse_timeframe(interval)

        if not is_alias:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"/analyze-scalp khusus XAUUSD/XAU/GOLD. Simbol '{symbol}' "
                             f"tidak didukung di sini -- pakai /analyze untuk crypto.",
                },
            )

        asset_class = "xau"

        # df Binance (PAXGUSDT) tetap diambil sbg fallback_df utk get_structure_df
        # -- SAMA seperti jalur /analyze, supaya sub-metode scalp jalan di atas
        # struktur XAUUSD asli (TwelveData) kalau tersedia, atau fallback yang
        # sama persis dengan mode swing kalau tidak (bukan sumber data lain lagi).
        df = fetch_klines(bsym, bint, limit=300)
        df_structure, structure_source = get_structure_df(asset_class, tf_label, df)

        live_price = fetch_live_price(bsym)
        spot_price, spot_label = fetch_spot_crosscheck(bsym)

        open_positions = {"LONG": open_long, "SHORT": open_short}
        result = detect_scalp_cascade(df_structure, lookback_candles=SCALP_LOOKBACK_CANDLES,
                                       open_positions=open_positions)
        setup_available = result.get("sub_method") is not None

        last_row = df_structure.iloc[-1]
        last_structure_time = (
            last_row["open_time"].isoformat() if "open_time" in df_structure.columns else None
        )

        return {
            "symbol": display,
            "interval": tf_label,
            "asset_class": asset_class,
            "mode": "scalp_cascade",
            "structure_source": structure_source,
            "last_structure_candle_time": last_structure_time,
            "last_structure_close": round(float(last_row["close"]), 4),
            "price_live": live_price,
            "price_is_live": live_price is not None,
            # spot_price XAUUSD asli (gold-api) sbg cross-check terhadap entry
            # yg dihitung dari df_structure -- sama filosofinya dgn /analyze.
            "spot_price": spot_price,
            "spot_label": spot_label,
            "setup_available": setup_available,
            **result,
            "guard_sl_pips_range": [SCALP_SL_MIN_PIPS, SCALP_SL_MAX_PIPS],
            "pip_size_assumed": SCALP_PIP_SIZE,
            "lookback_candles": SCALP_LOOKBACK_CANDLES,
            "concurrency": {
                "open_long_reported": open_long,
                "open_short_reported": open_short,
                "max_same_direction": SCALP_MAX_CONCURRENT_SAME_DIRECTION,
                "blocked_by_cap": result.get("blocked_by_concurrency_cap", False),
            },
            "calibration_status": "UNCALIBRATED -- parameter desain awal, belum divalidasi backtest "
                                   "per sub-metode. Gunakan hasil endpoint ini utk mengumpulkan data "
                                   "forward-test (demo) + backtest historis, bukan eksekusi live.",
            "instruction_for_assistant": (
                "SEBELUM memanggil endpoint ini, TANYAKAN dulu ke user berapa posisi LONG dan SHORT "
                "dari sistem ini yang MASIH TERBUKA sekarang di MT5, dan kirim sebagai open_long/"
                "open_short -- server tidak punya cara lain untuk tahu ini (eksekusi manual, tidak "
                "ada koneksi broker). Kalau setup_available=false DAN concurrency.blocked_by_cap=true, "
                "sampaikan ke user bahwa sinyal valid ada tapi ditahan karena cap posisi searah sudah "
                "penuh -- JANGAN sarankan buka posisi baru arah itu. Kalau setup_available=false karena "
                "alasan lain, sampaikan 'reason' apa adanya, JANGAN membuat atau menebak entry/SL/TP "
                "sendiri. Kalau setup_available=true, tetap tampilkan calibration_status ke user -- ini "
                "skeleton yang belum dikalibrasi, bukan sinyal final."
            ),
        }

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Scalp cascade gagal: {e}"})


if __name__ == "__main__":
    # Untuk test lokal saja. Di Railway, uvicorn yang menjalankan `app` (lihat start command).
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
