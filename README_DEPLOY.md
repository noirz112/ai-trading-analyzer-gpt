# Custom GPT + Action: AI Trading Analysis

Backend API siap-deploy untuk dipakai oleh sebuah **Custom GPT** lewat fitur *Actions*.
ChatGPT memanggil API ini (server-to-server, HTTPS), mengambil hasil analisis
kuantitatif (Entry / SL / TP / Confidence), lalu **menarasikan-nya** dalam bahasa
natural kepada Anda. Tidak ada eksekusi order — analysis-only.

## Arsitektur singkat

```
Anda: "analisa XAUUSD H1"
   │
   ▼
ChatGPT (Custom GPT)  ──Action GET /analyze──▶  Backend API (file ini)  ──▶  Binance/gold API
   │                                                  │
   ◀── narasi natural (ChatGPT bikin) ──────────────── ▼ JSON {entry,sl,tp,conf,reasons}
Anda: dapat setup + alasan dalam bahasa manusia
```

Keuntungan vs file HTML: **geo-block Binance tidak masalah** (server di region yang
diizinkan), dan ChatGPT sendiri yang jadi "LLM penjelas" — jadi gratis, tanpa API key LLM.

---

## Bagian A — Deploy backend (gratis, ~5 menit)

Direkomendasikan: **Render** (HTTPS otomatis, free tier cukup untuk pakai sesekali).

### Langkah

1. **Push folder `gpt_server/` ke GitHub** (repo baru, public/private bebas).
   File yang dibutuhkan: `server.py`, `engine.py`, `requirements.txt`.

2. **Buat Web Service di Render**:
   - Buka https://dashboard.render.com → *New +* → **Web Service**
   - Hubungkan repo GitHub Anda
   - Pengaturan:
     - **Name**: `trading-analyzer` (bebas)
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free
   - **Environment Variables**: tambah satu:
     - `BASE_URL` = `https://trading-analyzer.onrender.com`  ← ganti sesuai nama service Anda

3. **Deploy** → tunggu ~2 menit. Cek tab "Logs" sampai muncul:
   `Uvicorn running on http://0.0.0.0:port`

4. **Tes endpoint** (buka di browser atau curl):
   ```
   https://trading-analyzer.onrender.com/health
   https://trading-analyzer.onrender.com/analyze?symbol=XAUUSD&timeframe=H1
   ```
   Harus balas JSON dengan `direction`, `entry`, `stop_loss`, dll.

> **Catatan free tier Render:** service sleep setelah 15 menit idle, request
> pertama setelah idle butuh ~30 detik untuk "wake". Cukup untuk pemakaian analisis
> sesekali. Kalau mau selalu hidup, naik ke Starter ($7/bln) atau pakai Railway/Fly.io.

### Alternatif deploy cepat (tanpa GitHub)
- **Railway**: https://railway.app → *Deploy from GitHub* atau *Deploy from template*
- **Fly.io**: `fly launch` (perlu `flyctl`)
- **Render viaZIP**: tidak didukung; harus via repo Git.

---

## Bagian B — Buat Custom GPT-nya

1. Buka https://chat.openai.com/gpts (atae `chatgpt.com/gpts`) → **Create** (butuh Plus/Team).

2. Tab **Configure**:
   - **Name**: `AI Trading Analyst` (bebas)
   - **Description**: `Analisa Entry/SL/TP untuk crypto & emas, analysis-only.`
   - **Instructions** (copy teks di bagian C bawah ini)

3. Scroll ke bawah → **Actions** → **Create new action**:
   - **Import from URL**: tempel
     ```
     https://trading-analyzer.onrender.com/openapi.json
     ```
     (ganti domain sesuai nama service Render Anda)
   - Klik **Import**. Akan terdeteksi 3 action: `health`, `analyze`, `analyze_multi`.
   - Kalau diminta *server URL*, isi: `https://trading-analyzer.onrender.com`

4. **Save** Custom GPT. Selesai.

5. **Tes**: di chat Custom GPT ketik:
   - "analisa BTCUSDT H1"
   - "berapa setup XAUUSD D1?"
   - "bandingkan ETH M15, H1, H4"  ← ini akan memanggil `analyze_multi`

---

## Bagian C — Teks Instructions untuk Custom GPT

Copy blok berikut ke field **Instructions** Custom GPT Anda:

```
Anda adalah AI Trading Analyst. Tugas: memberi analisis trading (Entry, Stop Loss,
Take Profit, Confidence, alasan) UNTUK KEPENTINGAN ANALISIS SAJA. Anda TIDAK pernah
mengeksekusi order — semua eksekusi tetap manual oleh user.

Ketika user menyebut aset dan timeframe, panggil action "analyze" dengan:
  - symbol: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XAUUSD (emas), XAU, GOLD, dll.
  - timeframe: M1, M5, M15, M30, H1, H4, D1
  - rr (opsional, default 2): target risk:reward

Jika user meminta "bandingkan beberapa timeframe" / "confluence" / "konfirmasi
multi-timeframe", panggil action "analyze_multi" (otomatis M15+H1+H4).

Cara mempresentasikan hasil (gaya singkat & terstruktur):
1. Satu baris headline: ARAH (LONG/SHORT/NEUTRAL) + Confidence%.
2. Tabel level: Entry | Stop Loss | TP1 | TP2 | TP3 | Risk:Reward.
3. Bullet alasan teknikal (dari field reasons) — rangkum, jangan dump mentah.
4. Untuk emas (XAUUSD), sebutkan selisih proxy vs spot (field spot_reference).
5. Untuk analyze_multi, tekankan confluence: kalau 3 timeframe searah, setup kuat.
6. Selalu akhiri dengan disclaimer: "Analysis only, bukan saran keuangan. DYOR."

Aturan penting:
- Jika direction = NEUTRAL, JANGAN dipaksakan jadi trade. Katakan "tidak ada setup
  jelas, tunggu konfirmasi" dan sebutkan kenapa (skor di rentang -20..+20).
- Jangan mengarang angka. Hanya pakai data dari action. Kalau ada field null/missing,
  sebutkan.
- Confidence tinggi ≠ jaminan profit. Ingatkan manajemen risiko & position sizing.
- Bahasa: ikuti bahasa user (default Indonesia).
```

---

## Bagian D — Troubleshooting

| Gejala | Solusi |
|--------|--------|
| Action gagal / 500 saat di Custom GPT | Cek Render Logs. Kemungkinan BASE_URL belum diset → openapi.json tanpa `servers`. |
| `openapi.json` tidak ter-import | Pastikan URL pakai domain Render Anda yang sudah live. |
| Hasil `direction=NEUTRAL` terus | Bukan bug — market sedang sideways di TF itu. Coba TF lebih besar (H4/D1). |
| Render sleep / lambat di request pertama | Free tier. Tunggu ~30s, atau upgrade. |
| ChatGPT minta "server URL" | Isi `https://<nama>.onrender.com`. |
| Mau ganti region server (menghindari geo-block) | Render region bisa dipilih saat create. Pilih Frankfurt/Singapore. |

---

## File dalam folder ini

| File | Fungsi |
|------|--------|
| `server.py` | FastAPI app: endpoint `/analyze`, `/analyze_multi`, `/health` + OpenAPI custom. |
| `engine.py` | Engine analisis (indikator + scoring + setup builder) — sama dengan CLI. |
| `requirements.txt` | Dependency Python untuk deploy. |

## Endpoint lengkap (OpenAPI)

- `GET /health` → status server
- `GET /analyze?symbol=XAUUSD&timeframe=H1&rr=2` → analisis 1 simbol/TF
- `GET /analyze_multi?symbol=BTCUSDT&rr=2` → confluence M15+H1+H4
- `GET /openapi.json` → schema untuk Custom GPT (sudah memuat field `servers`)
