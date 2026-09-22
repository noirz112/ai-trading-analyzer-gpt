#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — Web API wrapper untuk engine.py
=============================================================================
Kenapa file ini dibutuhkan:
  engine.py adalah CLI script (pakai argparse, wajib diberi argumen symbol +
  interval). Kalau di-deploy ke Railway sebagai "Service" tanpa wrapper ini,
  Railway menjalankan proses TANPA argumen -> argparse langsung error dan
  proses exit(2) sedetik setelah start -> Railway tandai CRASHED -> restart
  otomatis -> crash lagi (loop). Itu penyebab error 502 yang muncul di GPT.

  server.py ini membuat engine.py tetap "hidup" sebagai web server yang
  listen di $PORT (wajib untuk Railway), dan dipanggil oleh Custom GPT lewat
  HTTP GET, bukan command line.

ENDPOINT:
  GET /health
      -> cek server hidup atau tidak

  GET /analyze?symbol=XAUUSD&interval=H1&rr=2&chart=false&cot=false&dxy_bias=neutral
      -> menjalankan analisa engine.py, return JSON berisi:
         - report      : teks laporan lengkap (persis output CLI)
         - direction, entry, sl, tp1, tp2, tp3, conf, score  : field terstruktur
         - chart_url   : link PNG chart (kalau chart=true)

DEPLOY DI RAILWAY:
  1. Pastikan requirements.txt berisi: pandas numpy requests matplotlib flask gunicorn
  2. Start command / Procfile:
       web: gunicorn server:app --bind 0.0.0.0:$PORT --timeout 90 --workers 1
  3. Jangan pakai "python engine.py ..." sebagai start command lagi.
"""
import io
import os
import contextlib
import traceback

from flask import Flask, request, jsonify, send_from_directory

from engine import (
    resolve_symbol, parse_timeframe, fetch_klines, symbol_has_futures,
    analyze_crypto, analyze_xau, build_setup, fetch_spot_crosscheck,
    fetch_cot_gold, print_report, plot_chart,
)

app = Flask(__name__)

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHART_DIR, exist_ok=True)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/analyze")
def analyze():
    symbol = request.args.get("symbol", "").strip()
    interval = request.args.get("interval", "").strip()
    rr = request.args.get("rr", default=2.0, type=float)
    want_chart = request.args.get("chart", "false").lower() in ("1", "true", "yes")
    want_cot = request.args.get("cot", "false").lower() in ("1", "true", "yes")
    dxy_bias = request.args.get("dxy_bias") or None

    if not symbol or not interval:
        return jsonify(error="Parameter 'symbol' dan 'interval' wajib diisi. "
                              "Contoh: /analyze?symbol=XAUUSD&interval=H1"), 400

    try:
        display, bsym, is_alias = resolve_symbol(symbol)
        bint, tf_label = parse_timeframe(interval)
        asset_class = "xau" if is_alias else "crypto"

        df = fetch_klines(bsym, bint, limit=300)
        has_futures = symbol_has_futures(bsym)

        if asset_class == "crypto":
            a = analyze_crypto(df, bsym, has_futures)
        else:
            a = analyze_xau(df, bsym, has_futures, dxy_bias_override=dxy_bias)

        s = build_setup(a, rr=rr)
        spot_price, spot_label = fetch_spot_crosscheck(bsym)
        cot_data = fetch_cot_gold() if (want_cot and asset_class == "xau") else None

        # Tangkap teks laporan (persis seperti output CLI) tanpa print ke stdout server
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(display, bsym, tf_label, a, s, asset_class,
                         is_alias=is_alias, spot_price=spot_price,
                         spot_label=spot_label, cot=cot_data)
        report_text = buf.getvalue()

        chart_url = None
        if want_chart:
            try:
                fname = f"chart_{display}_{tf_label}_v2.png"
                fpath = os.path.join(CHART_DIR, fname)
                # plot_chart menyimpan ke working dir; arahkan ke CHART_DIR
                cwd = os.getcwd()
                os.chdir(CHART_DIR)
                try:
                    plot_chart(display, bsym, tf_label, df, a, s)
                finally:
                    os.chdir(cwd)
                chart_url = request.host_url.rstrip("/") + f"/charts/{fname}"
            except Exception as e:
                report_text += f"\n[chart gagal: {e}]"

        return jsonify(
            symbol=display,
            interval=tf_label,
            asset_class=asset_class,
            direction=s["direction"],
            entry=s["entry"],
            sl=s["sl"],
            tp1=s["tp1"],
            tp2=s["tp2"],
            tp3=s["tp3"],
            confidence=round(s["conf"], 1),
            score=a["score"],
            report=report_text,
            chart_url=chart_url,
        )

    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        # Jangan biarkan exception bikin proses mati -> selalu balas JSON,
        # supaya GPT dapat pesan error yang jelas, bukan 502.
        traceback.print_exc()
        return jsonify(error=f"Analisa gagal: {e}"), 500


@app.get("/charts/<path:fname>")
def serve_chart(fname):
    return send_from_directory(CHART_DIR, fname)


if __name__ == "__main__":
    # Untuk test lokal saja. Di Railway, gunicorn yang menjalankan `app`.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
