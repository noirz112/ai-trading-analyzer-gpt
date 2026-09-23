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
)

app = FastAPI(
    title="AI Trading Analyzer API",
    description="Backend untuk Custom GPT — analisa liquidity/structure-based XAU & crypto.",
    version="1.0.0",
    servers=[
        {"url": "https://ai-trading-analyzer-gpt-production.up.railway.app"},
    ],
)

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
):
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
        order_type_detail = classify_order_type(s["direction"], s["entry"], spot_price)

        # Filter konservatif tambahan (lihat evaluate_tradeable() di engine.py):
        # tradeable=True HANYA kalau tier A DAN struktur HTF tidak ranging.
        # TIDAK menyembunyikan setup Tier B/C -- laporan tetap tampil penuh,
        # ini cuma flag rekomendasi ekstra buat GPT/user memutuskan.
        tradeable_eval = evaluate_tradeable(a, s)
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
                fname = f"chart_{display}_{tf_label}_v2.png"
                cwd = os.getcwd()
                os.chdir(CHART_DIR)
                try:
                    plot_chart(display, bsym, tf_label, df, a, s)
                finally:
                    os.chdir(cwd)
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


if __name__ == "__main__":
    # Untuk test lokal saja. Di Railway, uvicorn yang menjalankan `app` (lihat start command).
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
