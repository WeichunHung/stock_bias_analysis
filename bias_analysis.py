#!/usr/bin/env python3
"""
台股乖離率分析
==============
使用方式：
  python bias_analysis.py          → 啟動互動式本地伺服器

安裝依賴套件：
  pip install pandas numpy requests

數據來源：
  股價日線資料  →  FinMind API（免費版每日 600 次）
"""

import json
import os
import threading
import time
import urllib.parse
import uuid
import webbrowser
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import requests


# ══════════════════════════════════════════════
# 0. 參數設定
# ══════════════════════════════════════════════
TOKEN = os.environ.get(
    "FINMIND_TOKEN",
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiaHVuZ3dlaTAxMTciLCJlbWFpbCI6Imh1bmd3ZWkwMTE3QGdtYWlsLmNvbSIsInRva2VuX3ZlcnNpb24iOjB9.WiSBnuSUAAcRzOmrscAbLPcsGJ5U5YcInseBxyQlYe8"
)
FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"
REQUEST_DELAY = 1.0
SERVER_PORT   = int(os.environ.get("PORT", 8766))
MA_PERIODS    = [20, 60, 120]


# ══════════════════════════════════════════════
# 1. FinMind 工具函式
# ══════════════════════════════════════════════
def _finmind_get(dataset: str, data_id: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "dataset":    dataset,
        "data_id":    data_id,
        "start_date": start,
        "end_date":   end,
        "token":      TOKEN,
    }
    resp = requests.get(FINMIND_BASE, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != 200:
        raise RuntimeError(f"FinMind [{dataset}] 錯誤：{body.get('msg')}")
    time.sleep(REQUEST_DELAY)
    return pd.DataFrame(body["data"])


def fetch_stock_name(stock_id: str) -> str:
    try:
        r = requests.get(FINMIND_BASE, params={
            "dataset": "TaiwanStockInfo",
            "data_id": stock_id,
            "token":   TOKEN,
        }, timeout=8)
        data = r.json().get("data", [])
        if data:
            return data[-1].get("stock_name", stock_id)
    except Exception:
        pass
    return stock_id


# ══════════════════════════════════════════════
# 2. 回檔 / 回漲機率計算（共用核心）
# ══════════════════════════════════════════════
PAIRED_WINDOWS = {"BIAS20": 10, "BIAS60": 20, "BIAS120": 40}


def _signal_prob(close: pd.Series,
                 bias_series: dict[str, pd.Series],
                 pct_thresholds: list[int],
                 move_pcts: list[int],
                 direction: str) -> dict:
    """
    共用計算核心。
    direction = "down"  觸發：BIAS 上穿高百分位，衡量後續最大跌幅
    direction = "up"    觸發：BIAS 下穿低百分位，衡量後續最大漲幅
    """
    close_arr = close.values
    close_idx = close.index
    result: dict = {}

    def pct_or_none(n: int, total: int):
        return round(n / total * 100, 1) if total > 0 else None

    for key, bias in bias_series.items():
        fwd = PAIRED_WINDOWS.get(key)
        if fwd is None:
            continue
        valid = bias.dropna()
        if len(valid) < 60:
            continue

        thresholds_data = []
        for pct in pct_thresholds:
            thresh_val = float(np.percentile(valid, pct))

            if direction == "down":          # 上穿高百分位
                mask      = bias > thresh_val
                cross     = mask & ~mask.shift(1).fillna(False)
            else:                            # 下穿低百分位
                mask      = bias < thresh_val
                cross     = mask & ~mask.shift(1).fillna(False)

            trigger_dates = bias.index[cross]
            move_n  = {m: 0 for m in move_pcts}
            valid_n = 0

            for tdate in trigger_dates:
                loc = close_idx.searchsorted(tdate)
                if loc >= len(close_idx) or close_idx[loc] != tdate:
                    continue
                ref     = close_arr[loc]
                f_close = close_arr[loc + 1 : loc + 1 + fwd]
                if ref <= 0 or len(f_close) < fwd:
                    continue
                valid_n += 1
                if direction == "down":
                    extreme = (f_close.min() - ref) / ref * 100   # 最大跌幅（負值）
                    for m in move_pcts:
                        if extreme <= -m:
                            move_n[m] += 1
                else:
                    extreme = (f_close.max() - ref) / ref * 100   # 最大漲幅（正值）
                    for m in move_pcts:
                        if extreme >= m:
                            move_n[m] += 1

            row: dict = {
                "pct":      pct,
                "bias_val": round(thresh_val, 2),
                "n_events": int(len(trigger_dates)),
            }
            for m in move_pcts:
                row[f"p{m}"] = pct_or_none(move_n[m], valid_n)
            thresholds_data.append(row)

        result[key] = {"paired_days": fwd, "thresholds": thresholds_data}

    return result


def compute_pullback_prob(close: pd.Series,
                          bias_series: dict[str, pd.Series]) -> dict:
    """BIAS 上穿高百分位後，配對窗口內出現 -3%/-5%/-10% 的機率"""
    return _signal_prob(close, bias_series,
                        pct_thresholds=[60, 70, 80, 90],
                        move_pcts=[3, 5, 10],
                        direction="down")


def compute_rebound_prob(close: pd.Series,
                         bias_series: dict[str, pd.Series]) -> dict:
    """BIAS 下穿低百分位後，配對窗口內出現 +3%/+5%/+10% 的機率"""
    return _signal_prob(close, bias_series,
                        pct_thresholds=[40, 30, 20, 10],
                        move_pcts=[3, 5, 10],
                        direction="up")


def compute_current_prob(close: pd.Series,
                         bias_series: dict[str, pd.Series],
                         stats_list: list) -> dict:
    """
    依據目前各 BIAS 的百分位位置，找出歷史上相似位置（±8pp 帶寬）
    的所有交易日，分別計算 5 / 10 / 20 日內的漲跌機率。
    """
    BAND           = 8
    MOVE_PCTS      = [3, 5, 10]
    FORWARD_WINDOWS = [5, 10, 20]
    close_arr      = close.values
    close_idx      = close.index
    stats_map      = {s["名稱"]: s for s in stats_list}
    result: dict   = {}

    for key, bias in bias_series.items():
        s = stats_map.get(key)
        if s is None:
            continue
        valid = bias.dropna()
        if len(valid) < 30:
            continue

        current_val = s["目前值"]
        current_pct = s["百分位"]

        lo_pct = max(0.0,   current_pct - BAND)
        hi_pct = min(100.0, current_pct + BAND)
        lo_val = float(np.percentile(valid, lo_pct))
        hi_val = float(np.percentile(valid, hi_pct))

        candidates = bias.index[(bias >= lo_val) & (bias <= hi_val)]

        # 每個窗口獨立計數（資料末端的樣本可能缺少較長窗口資料）
        down_n  = {f: {m: 0 for m in MOVE_PCTS} for f in FORWARD_WINDOWS}
        up_n    = {f: {m: 0 for m in MOVE_PCTS} for f in FORWARD_WINDOWS}
        valid_n = {f: 0 for f in FORWARD_WINDOWS}

        for tdate in candidates:
            loc = close_idx.searchsorted(tdate)
            if loc >= len(close_idx) or close_idx[loc] != tdate:
                continue
            ref = close_arr[loc]
            if ref <= 0:
                continue
            for fwd in FORWARD_WINDOWS:
                f_close = close_arr[loc + 1 : loc + 1 + fwd]
                if len(f_close) < fwd:          # 末端資料不足，跳過
                    continue
                valid_n[fwd] += 1
                max_drop = (f_close.min() - ref) / ref * 100
                max_rise = (f_close.max() - ref) / ref * 100
                for m in MOVE_PCTS:
                    if max_drop <= -m:
                        down_n[fwd][m] += 1
                    if max_rise >= m:
                        up_n[fwd][m] += 1

        def p(n: int, tot: int) -> float | None:
            return round(n / tot * 100, 1) if tot > 0 else None

        result[key] = {
            "current_val": current_val,
            "current_pct": round(current_pct, 1),
            "n_similar":   valid_n[5],          # 以最短窗口的樣本數代表
            "lo_pct":      round(lo_pct, 0),
            "hi_pct":      round(hi_pct, 0),
            "windows": {
                str(fwd): {
                    "n":    valid_n[fwd],
                    "down": {str(m): p(down_n[fwd][m], valid_n[fwd]) for m in MOVE_PCTS},
                    "up":   {str(m): p(up_n[fwd][m],   valid_n[fwd]) for m in MOVE_PCTS},
                }
                for fwd in FORWARD_WINDOWS
            },
        }

    return result


# ══════════════════════════════════════════════
# 3. 乖離率計算
# ══════════════════════════════════════════════
def compute_bias(stock_id: str, start: str, end: str) -> dict:
    print(f"📡 抓取 {stock_id} 股價資料（{start} ～ {end}）…")
    df = _finmind_get("TaiwanStockPrice", stock_id, start, end)
    if df.empty:
        raise ValueError(f"查無股價資料：{stock_id}")

    df["date"]  = pd.to_datetime(df["date"])
    df          = df.sort_values("date").drop_duplicates(subset="date").set_index("date")
    close       = df["close"].astype(float)

    if len(close) < 10:
        raise ValueError(f"資料筆數不足（{len(close)} 筆），請確認股票代碼或日期範圍")

    # 過濾異常價格：單日漲跌超過 25% 視為未最終確認的盤中資料，排除之
    daily_chg = close.pct_change().abs()
    bad_mask  = daily_chg > 0.25
    if bad_mask.any():
        bad_dates = close.index[bad_mask].strftime("%Y-%m-%d").tolist()
        print(f"  ⚠️  排除異常價格日期（單日變動>25%）：{bad_dates}")
        close = close[~bad_mask]

    print(f"✅ 取得 {len(close)} 個交易日資料（{close.index[0].date()} ～ {close.index[-1].date()}）")

    bias_series: dict[str, pd.Series] = {}
    stats_list  = []

    latest_close = float(close.iloc[-1])
    latest_date  = close.index[-1].strftime("%Y/%m/%d")

    for n in MA_PERIODS:
        key  = f"BIAS{n}"
        ma   = close.rolling(n).mean()
        bias = (close - ma) / ma * 100
        bias_series[key] = bias

        valid = bias.dropna()
        if valid.empty:
            continue

        current_val = float(valid.iloc[-1])
        pct_rank    = float((valid < current_val).mean() * 100)
        stats_list.append({
            "MA":     n,
            "名稱":   key,
            "目前值": round(current_val, 2),
            "均值":   round(float(valid.mean()), 2),
            "最小值": round(float(valid.min()), 2),
            "最大值": round(float(valid.max()), 2),
            "百分位": round(pct_rank, 1),
            "標準差": round(float(valid.std()), 2),
            "P10":    round(float(np.percentile(valid, 10)), 2),
            "P25":    round(float(np.percentile(valid, 25)), 2),
            "P75":    round(float(np.percentile(valid, 75)), 2),
            "P90":    round(float(np.percentile(valid, 90)), 2),
        })

    # 組成時序資料（全部共用日期軸）
    combined = pd.DataFrame(bias_series).dropna(how="all")
    dates    = [d.strftime("%Y/%m/%d") for d in combined.index]

    series_data = {}
    for key in combined.columns:
        series_data[key] = [
            round(float(v), 3) if pd.notna(v) else None
            for v in combined[key]
        ]

    # 收盤價與均線序列（對齊日期軸）
    close_aligned = close.reindex(combined.index)
    series_data["close"] = [
        round(float(v), 2) if pd.notna(v) else None
        for v in close_aligned
    ]
    for n in MA_PERIODS:
        ma_aligned = close.rolling(n).mean().reindex(combined.index)
        series_data[f"MA{n}"] = [
            round(float(v), 2) if pd.notna(v) else None
            for v in ma_aligned
        ]

    pullback_prob = compute_pullback_prob(close, bias_series)
    rebound_prob  = compute_rebound_prob(close, bias_series)
    current_prob  = compute_current_prob(close, bias_series, stats_list)

    return {
        "latest_close":  latest_close,
        "latest_date":   latest_date,
        "stats":         stats_list,
        "dates":         dates,
        "series":        series_data,
        "period_days":   len(close),
        "pullback_prob": pullback_prob,
        "rebound_prob":  rebound_prob,
        "current_prob":  current_prob,
    }


# ══════════════════════════════════════════════
# 3. 加權指數 BIAS
# ══════════════════════════════════════════════
def fetch_taiex_bias(start: str, end: str) -> dict:
    """抓取台股加權指數（TAIEX）並計算 BIAS20/60/120"""
    try:
        print("📡 抓取加權指數（TAIEX）資料…")
        df = _finmind_get("TaiwanStockPrice", "TAIEX", start, end)
        if df.empty:
            return {}

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset="date").set_index("date")
        close = df["close"].astype(float)

        # 過濾異常點
        bad = close.pct_change().abs() > 0.25
        if bad.any():
            close = close[~bad]

        bias_dict: dict[str, pd.Series] = {}
        for n in MA_PERIODS:
            ma = close.rolling(n).mean()
            bias_dict[f"BIAS{n}"] = (close - ma) / ma * 100

        combined = pd.DataFrame(bias_dict).dropna(how="all")
        series: dict = {}
        for key in combined.columns:
            series[key] = [
                round(float(v), 3) if pd.notna(v) else None
                for v in combined[key]
            ]
        series["close"] = [
            round(float(v), 0) if pd.notna(v) else None
            for v in close.reindex(combined.index)
        ]

        print(f"✅ 加權指數：{len(close)} 個交易日")
        return {
            "dates":  [d.strftime("%Y/%m/%d") for d in combined.index],
            "series": series,
        }
    except Exception as exc:
        print(f"  ⚠️  加權指數資料抓取失敗：{exc}")
        return {}


# ══════════════════════════════════════════════
# 4. 背景任務管理
# ══════════════════════════════════════════════
_jobs: dict = {}


def _run_analysis(job_id: str, stock_id: str, start: str, end: str) -> None:
    try:
        _jobs[job_id] = {"status": "running", "progress": "查詢股票名稱…"}
        stock_name = fetch_stock_name(stock_id)

        _jobs[job_id]["progress"] = "抓取股價資料…"
        result = compute_bias(stock_id, start, end)

        _jobs[job_id] = {
            "status": "done",
            "data": {
                "stock_id":   stock_id,
                "stock_name": stock_name,
                "period":     f"{start} ～ {end}",
                **result,
            },
        }
        print(f"✅ [{stock_id}] 分析完成")
    except Exception as exc:
        print(f"✖ [{stock_id}] 分析失敗：{exc}")
        _jobs[job_id] = {"status": "error", "message": str(exc)}


# ══════════════════════════════════════════════
# 4. 日期工具
# ══════════════════════════════════════════════
def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def _three_years_ago() -> str:
    d = date.today() - timedelta(days=365 * 3 + 1)
    return d.strftime("%Y-%m-%d")


# ══════════════════════════════════════════════
# 5. HTTP 處理器
# ══════════════════════════════════════════════
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if parsed.path == "/":
            self._html(_build_html())

        elif parsed.path == "/api/analyze":
            stock_id = params.get("stock", "").strip().upper()
            start    = params.get("start", _three_years_ago())
            end      = params.get("end",   _today())
            if not stock_id:
                self._json({"error": "請輸入股票代碼"}, 400)
                return
            job_id = str(uuid.uuid4())[:8]
            _jobs[job_id] = {"status": "pending", "progress": "排隊中…"}
            threading.Thread(
                target=_run_analysis,
                args=(job_id, stock_id, start, end),
                daemon=True,
            ).start()
            self._json({"job_id": job_id})

        elif parsed.path == "/api/status":
            job = _jobs.get(params.get("id", ""))
            self._json(job if job else {"error": "job 不存在"},
                       200 if job else 404)

        else:
            self.send_response(404)
            self.end_headers()


# ══════════════════════════════════════════════
# 6. HTML 頁面
# ══════════════════════════════════════════════
def _build_html() -> str:
    today = _today()
    start = _three_years_ago()
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>台股乖離率分析</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:#FAFAF8;color:#1A1A18;padding:20px 28px;max-width:980px;margin:auto;}}
.search-bar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  background:#F2F2EF;border-radius:10px;padding:14px 16px;margin-bottom:16px;}}
.search-bar label{{font-size:12px;color:#666660;white-space:nowrap;}}
.search-bar input{{border:.5px solid #DDDDD8;background:#fff;border-radius:6px;
  padding:6px 10px;font-size:13px;color:#1A1A18;outline:none;}}
.search-bar input:focus{{border-color:#888780;}}
#stockInput{{width:110px;font-weight:500;}}
#startInput,#endInput{{width:128px;}}
#analyzeBtn{{background:#1A1A18;color:#fff;border:none;border-radius:6px;
  padding:7px 18px;font-size:13px;cursor:pointer;transition:opacity .15s;}}
#analyzeBtn:hover{{opacity:.8;}}
#analyzeBtn:disabled{{opacity:.4;cursor:not-allowed;}}
.quick-btns{{display:flex;gap:6px;flex-wrap:wrap;}}
.qbtn{{border:.5px solid #DDDDD8;background:#fff;padding:4px 10px;border-radius:5px;
  cursor:pointer;font-size:11px;color:#555550;}}
.qbtn:hover{{background:#EEEEED;}}
#statusBar{{font-size:12px;color:#888780;margin-bottom:14px;min-height:18px;
  display:flex;align-items:center;gap:8px;}}
.spinner{{width:14px;height:14px;border:2px solid #E0E0DC;border-top-color:#888780;
  border-radius:50%;animation:spin .8s linear infinite;display:none;}}
@keyframes spin{{to{{transform:rotate(360deg);}}}}
h1{{font-size:17px;font-weight:600;margin-bottom:3px;}}
.sub{{font-size:12px;color:#888780;margin-bottom:18px;}}
.price-badge{{display:inline-block;background:#1A1A18;color:#fff;
  font-size:13px;padding:3px 10px;border-radius:5px;margin-left:10px;font-weight:500;}}

/* BIAS 卡片 */
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px;}}
@media(min-width:640px){{.cards{{grid-template-columns:repeat(6,1fr);}}}}
.card{{background:#fff;border:1px solid #E8E8E4;border-radius:10px;padding:12px 8px;
  text-align:center;cursor:pointer;transition:all .15s;}}
.card:hover,.card.active{{border-color:#1A1A18;box-shadow:0 0 0 1.5px #1A1A18;}}
.card-name{{font-size:11px;color:#888780;margin-bottom:5px;font-weight:500;}}
.card-val{{font-size:22px;font-weight:600;line-height:1.1;}}
.card-pct{{font-size:10px;color:#888780;margin-top:4px;}}
.card-bar{{height:4px;border-radius:2px;background:#EEE;margin-top:7px;overflow:hidden;}}
.card-bar-fill{{height:100%;border-radius:2px;transition:width .4s;}}

/* 統計表格 */
.stats-section{{margin-bottom:20px;}}
.stats-title{{font-size:12px;font-weight:500;color:#666660;letter-spacing:.04em;margin-bottom:8px;}}
.stats-table{{width:100%;border-collapse:collapse;font-size:12px;}}
.stats-table th{{text-align:left;padding:6px 10px;border-bottom:1px solid #E8E8E4;
  color:#888780;font-weight:500;white-space:nowrap;}}
.stats-table td{{padding:6px 10px;border-bottom:.5px solid #F0F0EC;white-space:nowrap;}}
.stats-table tr:last-child td{{border-bottom:none;}}
.stats-table tr:hover td{{background:#F8F8F5;}}
.range-bar{{position:relative;height:8px;background:#EEE;border-radius:4px;
  width:180px;display:inline-block;vertical-align:middle;}}
.range-bar-fill{{position:absolute;height:100%;border-radius:4px;}}
.range-dot{{position:absolute;top:50%;transform:translate(-50%,-50%);
  width:10px;height:10px;border-radius:50%;border:2px solid #fff;}}

/* 走勢圖 */
.chart-section{{margin-bottom:20px;}}
.chart-header{{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;}}
.chart-title{{font-size:12px;font-weight:500;color:#666660;letter-spacing:.04em;}}
.row-btns{{display:flex;gap:5px;flex-wrap:wrap;}}
.cbtn{{border:.5px solid #DDDDD8;background:transparent;padding:3px 10px;border-radius:5px;
  cursor:pointer;font-size:11px;color:#555550;transition:all .15s;}}
.cbtn.active{{background:#EEEEED;color:#1A1A18;border-color:#AAAAAA;font-weight:500;}}
.chart-wrap{{position:relative;width:100%;}}
.legend-row{{display:flex;gap:14px;font-size:11px;color:#666660;margin-bottom:6px;flex-wrap:wrap;}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle;}}
hr{{border:none;border-top:.5px solid #E5E5E2;margin:18px 0;}}
#dashboard{{display:none;}}
#emptyState{{text-align:center;padding:60px 0;color:#AAAAAA;font-size:14px;}}
.note{{font-size:11px;color:#AAA;margin-top:4px;}}
</style>
</head>
<body>

<div class="search-bar">
  <div style="width:100%;font-size:16px;font-weight:700;color:#333;margin-bottom:10px;letter-spacing:.04em;">個股熱度分析</div>
  <label>股票代碼</label>
  <input id="stockInput" type="text" placeholder="如 2330" maxlength="6" autocomplete="off">
  <label>起始日</label>
  <input id="startInput" type="date" value="{start}">
  <label>結束日</label>
  <input id="endInput" type="date" value="{today}">
  <button id="analyzeBtn" onclick="startAnalysis()">分析</button>
  <div class="quick-btns">
    <button class="qbtn" onclick="setStock('2330')">台積電</button>
    <button class="qbtn" onclick="setStock('2317')">鴻海</button>
    <button class="qbtn" onclick="setStock('2454')">聯發科</button>
    <button class="qbtn" onclick="setStock('2308')">台達電</button>
    <button class="qbtn" onclick="setStock('2382')">廣達</button>
    <button class="qbtn" onclick="setStock('006208')">富邦台50</button>
  </div>
</div>

<div id="statusBar">
  <div class="spinner" id="spinner"></div>
  <span id="statusMsg">輸入股票代碼後點擊「分析」</span>
</div>

<div id="emptyState">📈 在上方輸入股票代碼開始分析乖離率</div>

<div id="dashboard">
  <h1 id="pageTitle"></h1>
  <div class="sub" id="pageSub"></div>

  <!-- 六個 MA 卡片 -->
  <div class="cards" id="biasCards"></div>

  <!-- 歷史統計表格 -->
  <div class="stats-section">
    <div class="stats-title">歷史統計區間（近3年）</div>
    <table class="stats-table" id="statsTable">
      <thead><tr>
        <th>指標</th><th>目前值</th><th>百分位</th>
        <th>最小值</th><th>P10</th><th>P25</th><th>均值</th><th>P75</th><th>P90</th><th>最大值</th>
        <th style="width:200px">歷史分布位置</th>
      </tr></thead>
      <tbody id="statsBody"></tbody>
    </table>
  </div>

  <!-- 走勢圖 -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">乖離率走勢圖</span>
      <div class="row-btns" id="chartBtns"></div>
      <div class="row-btns" style="margin-left:auto;">
        <button class="cbtn active" onclick="setRange(90)">3M</button>
        <button class="cbtn" onclick="setRange(180)">6M</button>
        <button class="cbtn" onclick="setRange(365)">1Y</button>
        <button class="cbtn" onclick="setRange(0)">全部</button>
      </div>
    </div>
    <div class="legend-row" id="chartLegend"></div>
    <div class="chart-wrap" style="height:270px;"><canvas id="biasChart"></canvas></div>
  </div>

  <!-- 股價均線圖 -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">股價與均線</span>
      <div class="legend-row" id="taixLegend" style="margin-left:16px;"></div>
    </div>
    <div class="chart-wrap" style="height:220px;"><canvas id="histChart"></canvas></div>
  </div>

  <!-- 目前乖離率即時訊號 -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">目前乖離率的漲跌機率</span>
      <span style="font-size:11px;color:#AAA;margin-left:8px;">依當前百分位位置，查歷史相似樣本的後續漲跌分布</span>
    </div>
    <div id="currentProbSection"></div>
  </div>

  <!-- 回檔機率分析 -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">乖離率觸發後的回檔機率（歷史統計）</span>
      <span style="font-size:11px;color:#AAA;margin-left:8px;">BIAS 首次上穿高百分位後，N 個交易日內出現指定跌幅的機率</span>
    </div>
    <div id="pullbackSection"></div>
  </div>

  <!-- 回漲機率分析 -->
  <div class="chart-section">
    <div class="chart-header">
      <span class="chart-title">乖離率觸發後的回漲機率（歷史統計）</span>
      <span style="font-size:11px;color:#AAA;margin-left:8px;">BIAS 首次下穿低百分位（超賣）後，N 個交易日內出現指定漲幅的機率</span>
    </div>
    <div id="reboundSection"></div>
  </div>

  <hr>
  <div style="font-size:11px;color:#AAA;">
    乖離率 (BIAS) = (收盤價 − N日均線) ÷ N日均線 × 100%<br>
    觸發定義：乖離率從下方首次穿越百分位閾值（上穿）｜回檔定義：觸發後 N 日內最低收盤相對觸發日收盤的跌幅<br>
    數據來源：FinMind API　分析期間：近3年日線資料
  </div>
</div>

<script>
const MA_COLORS = {{
  "BIAS20": "#D85A30",
  "BIAS60": "#7F77DD",
  "BIAS120":"#1D9E75",
}};
const MA_KEYS = Object.keys(MA_COLORS);

let DATA = null, selMA = "BIAS20", chartInst = null, histInst = null;
let allDates = [], allSeries = {{}};
let displayDays = 90;

function setStock(code) {{
  document.getElementById('stockInput').value = code;
  startAnalysis();
}}

function startAnalysis() {{
  const stock = document.getElementById('stockInput').value.trim();
  const start = document.getElementById('startInput').value;
  const end   = document.getElementById('endInput').value;
  if (!stock) {{ alert('請輸入股票代碼'); return; }}
  [chartInst, histInst].forEach(c => c && c.destroy());
  chartInst = histInst = null;
  document.getElementById('dashboard').style.display = 'none';
  document.getElementById('emptyState').style.display = 'none';
  setStatus('loading', '送出分析請求…');
  document.getElementById('analyzeBtn').disabled = true;

  fetch(`/api/analyze?stock=${{encodeURIComponent(stock)}}&start=${{start}}&end=${{end}}`)
    .then(r => r.json())
    .then(res => {{
      if (res.error) {{ setStatus('error', res.error); return; }}
      poll(res.job_id);
    }})
    .catch(e => setStatus('error', '請求失敗：' + e));
}}

let pollTimer = null;
function poll(jobId) {{
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {{
    fetch(`/api/status?id=${{jobId}}`).then(r => r.json()).then(job => {{
      if (job.status === 'running') {{
        setStatus('loading', job.progress || '分析中…');
      }} else if (job.status === 'done') {{
        clearInterval(pollTimer);
        setStatus('done', '');
        renderDashboard(job.data);
        document.getElementById('analyzeBtn').disabled = false;
      }} else if (job.status === 'error') {{
        clearInterval(pollTimer);
        setStatus('error', '分析失敗：' + job.message);
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('emptyState').style.display = 'block';
      }}
    }}).catch(() => {{}});
  }}, 1500);
}}

function setStatus(type, msg) {{
  document.getElementById('spinner').style.display = type === 'loading' ? 'block' : 'none';
  const t = document.getElementById('statusMsg');
  t.textContent = msg;
  t.style.color = type === 'error' ? '#E24B4A' : '#888780';
}}

function biasColor(val, p10, p25, p75, p90) {{
  if (val >= p90) return '#E24B4A';   // 超強正乖離
  if (val >= p75) return '#EF9F27';   // 偏正乖離
  if (val <= p10) return '#1D9E75';   // 超強負乖離（低估）
  if (val <= p25) return '#5DCAA5';   // 偏負乖離
  return '#888780';
}}

function renderDashboard(data) {{
  DATA = data;
  allDates  = data.dates;
  allSeries = data.series;

  // 每次分析完畢，預設顯示全部資料並同步按鈕
  displayDays = 0;
  document.querySelectorAll('.row-btns .cbtn[onclick]').forEach(b => {{
    b.classList.toggle('active', b.getAttribute('onclick') === 'setRange(0)');
  }});

  document.getElementById('pageTitle').textContent =
    `${{data.stock_name}}（${{data.stock_id}}）乖離率分析`;
  const closeStr = data.latest_close ? `　收盤價 ${{data.latest_close.toLocaleString()}} 元` : '';
  document.getElementById('pageSub').textContent =
    `資料日期：${{data.latest_date}}${{closeStr}}　近3年共 ${{data.period_days}} 個交易日`;

  document.getElementById('dashboard').style.display = 'block';
  renderCards(data);
  renderStatsTable(data);
  renderChartBtns(data);
  renderCurrentProbCards(data);
  renderPullbackTable(data);
  renderReboundTable(data);
  requestAnimationFrame(() => {{
    renderBiasChart(data);
    renderTaixChart(data);
  }});
}}

function renderCards(data) {{
  const el = document.getElementById('biasCards');
  el.innerHTML = '';
  for (const s of data.stats) {{
    const key  = s["名稱"];
    const val  = s["目前值"];
    const pct  = s["百分位"];
    const p10  = s["P10"], p25 = s["P25"], p75 = s["P75"], p90 = s["P90"];
    const color = biasColor(val, p10, p25, p75, p90);
    const label = pct >= 90 ? '極度過熱' : pct >= 75 ? '貪婪' :
                  pct <= 10 ? '極度恐慌' : pct <= 25 ? '超賣' : '中性';
    const c = document.createElement('div');
    c.className = 'card' + (key === selMA ? ' active' : '');
    c.onclick = () => {{ selMA = key; selectCard(key, data); }};
    c.id = 'card_' + key;
    const fillPct = pct;
    c.innerHTML = `
      <div class="card-name">MA${{s["MA"]}}</div>
      <div class="card-val" style="color:${{color}}">${{val > 0 ? '+' : ''}}${{val.toFixed(2)}}%</div>
      <div class="card-pct">${{label}}（${{pct.toFixed(0)}}%分位）</div>
      <div class="card-bar"><div class="card-bar-fill" style="width:${{fillPct}}%;background:${{color}}"></div></div>
    `;
    el.appendChild(c);
  }}
}}

function selectCard(key, data) {{
  selMA = key;
  document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
  const c = document.getElementById('card_' + key);
  if (c) c.classList.add('active');
  document.querySelectorAll('#chartBtns .cbtn').forEach(b => {{
    b.classList.toggle('active', b.dataset.key === key);
  }});
}}

function renderStatsTable(data) {{
  const tbody = document.getElementById('statsBody');
  tbody.innerHTML = '';
  for (const s of data.stats) {{
    const val   = s["目前値"] ?? s["目前值"];
    const cur   = s["目前值"];
    const color = biasColor(cur, s["P10"], s["P25"], s["P75"], s["P90"]);
    const min   = s["最小值"], max = s["最大值"];
    const range = max - min || 1;
    const dotPct = ((cur - min) / range * 100).toFixed(1);
    const p10Pct = ((s["P10"] - min) / range * 100).toFixed(1);
    const p90Pct = ((s["P90"] - min) / range * 100).toFixed(1);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:500;color:#444">${{s["名稱"]}}</td>
      <td style="color:${{color}};font-weight:600">${{cur > 0 ? '+' : ''}}${{cur.toFixed(2)}}%</td>
      <td>
        <div style="display:inline-flex;align-items:center;gap:4px;">
          <div style="width:50px;height:6px;background:#EEE;border-radius:3px;position:relative;display:inline-block;vertical-align:middle;">
            <div style="position:absolute;height:100%;left:${{p10Pct}}%;width:${{(p90Pct-p10Pct).toFixed(1)}}%;background:#D0D0CC;border-radius:2px;"></div>
            <div style="position:absolute;top:50%;left:${{dotPct}}%;transform:translate(-50%,-50%);width:8px;height:8px;border-radius:50%;background:${{color}};border:1.5px solid #fff;box-shadow:0 0 0 1px ${{color}};"></div>
          </div>
          <span style="font-size:11px;color:#666">${{s["百分位"].toFixed(0)}}%</span>
        </div>
      </td>
      <td>${{s["最小值"].toFixed(2)}}%</td>
      <td style="color:#888">${{s["P10"].toFixed(2)}}%</td>
      <td style="color:#888">${{s["P25"].toFixed(2)}}%</td>
      <td>${{s["均值"].toFixed(2)}}%</td>
      <td style="color:#888">${{s["P75"].toFixed(2)}}%</td>
      <td style="color:#888">${{s["P90"].toFixed(2)}}%</td>
      <td>${{s["最大值"].toFixed(2)}}%</td>
      <td>
        <div class="range-bar">
          <div class="range-bar-fill" style="left:${{p10Pct}}%;width:${{(p90Pct-p10Pct).toFixed(1)}}%;background:#E0E0DC;"></div>
          <div class="range-dot" style="left:${{dotPct}}%;background:${{color}};"></div>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  }}
}}

function renderChartBtns(data) {{
  const el = document.getElementById('chartBtns');
  el.innerHTML = '';
  for (const s of data.stats) {{
    const key = s["名稱"];
    const b = document.createElement('button');
    b.className = 'cbtn' + (key === selMA ? ' active' : '');
    b.dataset.key = key;
    b.textContent = key;
    b.style.borderColor = MA_COLORS[key] + '80';
    b.onclick = () => {{ selectCard(key, data); }};
    el.appendChild(b);
  }}
}}

function setRange(days) {{
  displayDays = days;
  document.querySelectorAll('.row-btns .cbtn[onclick]').forEach(b => {{
    b.classList.toggle('active', b.getAttribute('onclick') === `setRange(${{days}})`);
  }});
  if (chartInst) {{ chartInst.destroy(); chartInst = null; }}
  if (histInst)  {{ histInst.destroy();  histInst  = null; }}
  renderBiasChart(DATA);
  renderTaixChart(DATA);
}}

function sliceTail(arr, days) {{
  if (!days || days >= arr.length) return arr;
  return arr.slice(-days);
}}

function renderBiasChart(data) {{
  if (chartInst) chartInst.destroy();
  const TC = "rgba(0,0,0,0.35)", GC = "rgba(0,0,0,0.06)";
  const dates  = sliceTail(data.dates, displayDays);
  const nSlice = dates.length;
  const keyList = data.stats.map(s => s["名稱"]).filter(k => data.series[k]);

  // BIAS 線（左軸）
  const datasets = keyList.map(key => {{
    const raw = data.series[key] || [];
    return {{
      type: "line",
      label: key,
      data: raw.slice(-nSlice),
      borderColor: MA_COLORS[key],
      borderWidth: key === selMA ? 2.5 : 1,
      pointRadius: 0,
      tension: 0.2,
      yAxisID: "yL",
    }};
  }});

  // 零軸
  datasets.push({{
    type: "line",
    label: "_zero",
    data: Array(nSlice).fill(0),
    borderColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderDash: [5, 4],
    pointRadius: 0,
    yAxisID: "yL",
  }});

  // 股價折線（右軸）
  const closeRaw = data.series["close"] || [];
  const closeVals = closeRaw.slice(-nSlice);
  datasets.push({{
    type: "line",
    label: "股價",
    data: closeVals,
    borderColor: "#1A1A18",
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1,
    fill: false,
    yAxisID: "yR",
    order: -1,
  }});

  // 圖例
  const legend = document.getElementById('chartLegend');
  legend.innerHTML =
    keyList.map(k =>
      `<span><span class="dot" style="background:${{MA_COLORS[k]}}"></span>${{k}}</span>`
    ).join('') +
    `<span><span class="dot" style="background:#1A1A18"></span>股價（右軸）</span>`;

  chartInst = new Chart(document.getElementById('biasChart'), {{
    data: {{ labels: dates, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: items => items[0].label,
            label: ctx => {{
              if (ctx.dataset.label === "_zero") return "";
              const v = ctx.parsed.y;
              if (v == null) return "";
              if (ctx.dataset.label === "股價")
                return ` 股價：${{v.toLocaleString()}} 元`;
              return ` ${{ctx.dataset.label}}：${{v >= 0 ? "+" : ""}}${{v.toFixed(2)}}%`;
            }},
            filter: item => item.dataset.label !== "_zero",
          }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: TC, font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }},
          grid: {{ color: GC }},
        }},
        yL: {{
          type: "linear",
          position: "left",
          ticks: {{
            color: TC, font: {{ size: 10 }},
            callback: v => (v >= 0 ? "+" : "") + v.toFixed(1) + "%",
          }},
          grid: {{ color: GC }},
          title: {{ display: true, text: "乖離率 (%)", color: TC, font: {{ size: 10 }} }},
        }},
        yR: {{
          type: "linear",
          position: "right",
          ticks: {{
            color: "#1A1A18", font: {{ size: 10, weight: "500" }},
            callback: v => v.toLocaleString(),
          }},
          grid: {{ display: false }},
          title: {{ display: true, text: "股價 (元)", color: "#1A1A18", font: {{ size: 10, weight: "500" }} }},
        }},
      }},
    }},
  }});
}}

function renderTaixChart(data) {{
  if (histInst) {{ histInst.destroy(); histInst = null; }}
  const s  = data.series;
  const TC = "rgba(0,0,0,0.4)", GC = "rgba(0,0,0,0.06)";
  const MA_KEYS   = ["MA20","MA60","MA120"];
  const MA_COLORS = {{"MA20":"#D85A30","MA60":"#7F77DD","MA120":"#1D9E75"}};

  // 更新圖例
  const legEl = document.getElementById('taixLegend');
  if (legEl) {{
    legEl.innerHTML =
      `<span><span class="dot" style="background:#1A1A18"></span>股價</span>` +
      MA_KEYS.map(k =>
        `<span><span class="dot" style="background:${{MA_COLORS[k]}}"></span>${{k}}</span>`
      ).join('');
  }}

  if (!s || !data.dates || !data.dates.length) {{
    const ctx = document.getElementById('histChart').getContext('2d');
    ctx.font = "13px sans-serif"; ctx.fillStyle = "#AAA";
    ctx.fillText("股價資料無法取得", 20, 60);
    return;
  }}

  const dates  = sliceTail(data.dates, displayDays);
  const nSlice = dates.length;

  // 股價（右軸，粗線）
  const datasets = [{{
    type: "line", label: "股價",
    data: (s["close"] || []).slice(-nSlice),
    borderColor: "#1A1A18", borderWidth: 2,
    pointRadius: 0, tension: 0.1,
    yAxisID: "yR", order: -1,
  }}];

  // MA 線（右軸）
  MA_KEYS.filter(k => s[k]).forEach(key => {{
    datasets.push({{
      type: "line", label: key,
      data: (s[key] || []).slice(-nSlice),
      borderColor: MA_COLORS[key],
      borderWidth: 1.5,
      pointRadius: 0, tension: 0.2,
      yAxisID: "yR",
    }});
  }});

  histInst = new Chart(document.getElementById('histChart'), {{
    data: {{ labels: dates, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: items => items[0].label,
            label: ctx => {{
              const v = ctx.parsed.y;
              if (v == null) return "";
              return ` ${{ctx.dataset.label}}：${{v.toLocaleString()}}`;
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: TC, font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }},
          grid: {{ color: GC }},
        }},
        yR: {{
          type: "linear", position: "right",
          ticks: {{ color: TC, font: {{ size: 10 }}, callback: v => v.toLocaleString() }},
          grid: {{ color: GC }},
        }},
      }},
    }},
  }});
}}

// ── 回檔機率表格 ──────────────────────────────
function probBg(p) {{
  if (p === null) return "#F5F5F2";
  if (p >= 70)   return "#FDECEA";
  if (p >= 50)   return "#FEF3E2";
  if (p >= 30)   return "#FEFCE2";
  return "#EAF5EA";
}}
function probFg(p) {{
  if (p === null) return "#BBB";
  if (p >= 70)   return "#C0392B";
  if (p >= 50)   return "#C55A11";
  if (p >= 30)   return "#9A7D0A";
  return "#1E8449";
}}

// ── 目前乖離率即時訊號卡片 ──────────────────────
function currentPctColor(pct) {{
  if (pct >= 90) return "#C0392B";
  if (pct >= 75) return "#C55A11";
  if (pct >= 50) return "#888";
  if (pct <= 10) return "#1E8449";
  if (pct <= 25) return "#239B56";
  return "#888";
}}

function renderCurrentProbCards(data) {{
  const el = document.getElementById('currentProbSection');
  if (!el || !data.current_prob) return;

  const KEYS    = ["BIAS20","BIAS60","BIAS120"];
  const COLORS  = {{"BIAS20":"#D85A30","BIAS60":"#7F77DD","BIAS120":"#1D9E75"}};
  const FWDS      = ["5","10","20"];
  const MOVES     = ["3","5","10"];
  const DOWN_MOVES = ["10","5","3"];  // 由大到小排列

  // 表格 header cell
  const th = (txt, extra='') =>
    `<th style="padding:5px 6px;border:1px solid #E8E8E4;background:#F5F5F2;
       font-size:10px;font-weight:500;white-space:nowrap;text-align:center;
       color:#666;${{extra}}">${{txt}}</th>`;

  // 數值 td（down=紅色調，up=綠色調）
  const td_prob = (p, dir) => {{
    if (p === null) return `<td style="padding:5px 4px;text-align:center;border:1px solid #E8E8E4;color:#CCC;">—</td>`;
    const bg = dir === 'down' ? probBg(p)    : reboundBg(p);
    const fg = dir === 'down' ? probFg(p)    : reboundFg(p);
    return `<td style="padding:5px 4px;text-align:center;border:1px solid #E8E8E4;
      background:${{bg}};color:${{fg}};font-size:11px;font-weight:700;">${{p.toFixed(0)}}%</td>`;
  }};

  // 觀察天數列頭 td
  const td_row = (txt) =>
    `<td style="padding:5px 7px;border:1px solid #E8E8E4;background:#FAFAF8;
       font-size:11px;font-weight:600;color:#555;white-space:nowrap;">${{txt}}</td>`;

  let html = `<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;">`;

  for (const key of KEYS) {{
    const d = data.current_prob[key];
    if (!d) continue;
    const color    = COLORS[key];
    const valStr   = (d.current_val >= 0 ? '+' : '') + d.current_val.toFixed(2) + '%';
    const pctColor = currentPctColor(d.current_pct);

    // 計算多空分數：各窗口 (上漲3/5/10% 平均) - (下跌3/5/10% 平均)，再取 5/10/20日 平均
    let scoreSum = 0, scoreCnt = 0;
    FWDS.forEach(f => {{
      const w = d.windows && d.windows[f];
      if (!w) return;
      const upAvg  = (["3","5","10"].map(m => w.up[m]  ?? 0).reduce((a,b)=>a+b,0)) / 3;
      const downAvg= (["3","5","10"].map(m => w.down[m] ?? 0).reduce((a,b)=>a+b,0)) / 3;
      scoreSum += (upAvg - downAvg); scoreCnt++;
    }});
    const netScore = scoreCnt > 0 ? scoreSum / scoreCnt : null;
    let signal, sigColor, sigBg, sigIcon;
    if (netScore === null)      {{ signal='資料不足'; sigColor='#AAA';    sigBg='#F5F5F2'; sigIcon='–'; }}
    else if (netScore >  25)   {{ signal='強烈偏多'; sigColor='#C0392B'; sigBg='#FDECEA'; sigIcon='▲▲'; }}
    else if (netScore >   8)   {{ signal='偏多';     sigColor='#E24B4A'; sigBg='#FEF5F5'; sigIcon='▲'; }}
    else if (netScore < -25)   {{ signal='強烈偏空'; sigColor='#1E8449'; sigBg='#EAF5EA'; sigIcon='▼▼'; }}
    else if (netScore <  -8)   {{ signal='偏空';     sigColor='#27AE60'; sigBg='#F0FAF0'; sigIcon='▼'; }}
    else                       {{ signal='中性';     sigColor='#888';    sigBg='#F5F5F2'; sigIcon='◆'; }}
    // 儀表條：將 netScore 從 [-50,+50] 映射到 [0%,100%]
    const meterPct = netScore !== null ? Math.min(96, Math.max(4, (netScore + 50))) : 50;
    const meterColor = netScore > 0 ? '#C0392B' : netScore < 0 ? '#27AE60' : '#BBB';

    html += `
    <div style="flex:1;min-width:280px;background:#fff;border:1px solid #E8E8E4;
                border-radius:10px;padding:14px 16px;border-top:3px solid ${{color}};">
      <!-- 標題列 -->
      <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:10px;gap:10px;">
        <div>
          <div style="font-size:13px;font-weight:700;color:${{color}};margin-bottom:4px;">${{key}}</div>
          <div style="font-size:15px;font-weight:700;color:${{pctColor}};line-height:1.1;">${{valStr}}</div>
          <div style="font-size:10px;background:${{pctColor}}18;color:${{pctColor}};
                      padding:2px 7px;border-radius:4px;font-weight:600;margin-top:3px;display:inline-block;">
            歷史 ${{d.current_pct.toFixed(0)}}% 分位
          </div>
        </div>
        <!-- 多空概況 -->
        <div style="flex-shrink:0;text-align:center;min-width:80px;">
          <div style="font-size:18px;font-weight:800;color:${{sigColor}};
                      background:${{sigBg}};border-radius:8px;padding:6px 10px;
                      border:1.5px solid ${{sigColor}}30;line-height:1.2;">
            ${{sigIcon}}<br>
            <span style="font-size:11px;">${{signal}}</span>
          </div>
          <div style="margin-top:6px;position:relative;height:10px;">
            <div style="position:absolute;top:4px;left:0;right:0;height:2px;background:#EEE;border-radius:2px;"></div>
            <div style="position:absolute;top:0;left:${{meterPct}}%;transform:translateX(-50%);
                        width:10px;height:10px;border-radius:50%;
                        background:${{meterColor}};border:2px solid #fff;box-shadow:0 0 0 1.5px ${{meterColor}};"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:8px;color:#CCC;margin-top:3px;">
            <span>偏空</span><span>偏多</span>
          </div>
        </div>
      </div>
      <div style="font-size:10px;color:#AAA;margin-bottom:8px;padding-bottom:7px;border-bottom:1px solid #F0F0EC;">
        相似樣本（P${{d.lo_pct|0}}–P${{d.hi_pct|0}}）
      </div>
      <!-- 機率表格 -->
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr>
          ${{th('觀察','text-align:left;')}}
          ${{th('n')}}
          ${{DOWN_MOVES.map(m => th(`-${{m}}%`,'color:#C0392B;')).join('')}}
          ${{MOVES.map(m => th(`+${{m}}%`,'color:#27AE60;')).join('')}}
        </tr></thead>
        <tbody>
        ${{FWDS.map(f => {{
          const w = d.windows && d.windows[f];
          if (!w) return `<tr>${{td_row(f+'日')}}<td colspan="7" style="text-align:center;color:#CCC;border:1px solid #E8E8E4;font-size:10px;">—</td></tr>`;
          return `<tr>
            ${{td_row(f+'日')}}
            <td style="padding:5px 4px;text-align:center;border:1px solid #E8E8E4;font-size:10px;color:#999;">${{w.n}}</td>
            ${{DOWN_MOVES.map(m => td_prob(w.down[m], 'down')).join('')}}
            ${{MOVES.map(m => td_prob(w.up[m],   'up')).join('')}}
          </tr>`;
        }}).join('')}}
        </tbody>
      </table>
    </div>`;
  }}

  html += `</div>`;

  el.innerHTML = html;
}}

// ── 共用表格建構器 ─────────────────────────────
function _buildProbTable(sectionId, probData, direction) {{
  const el = document.getElementById(sectionId);
  if (!el || !probData) return;

  const KEYS    = ["BIAS20","BIAS60","BIAS120"];
  const COLORS  = {{"BIAS20":"#D85A30","BIAS60":"#7F77DD","BIAS120":"#1D9E75"}};
  const isDown  = direction === "down";
  const moves   = isDown ? [3,5,10] : [3,5,10];
  const labels  = isDown
    ? ["跌 -3%","跌 -5%","跌 -10%"]
    : ["漲 +3%","漲 +5%","漲 +10%"];
  const bgFn    = isDown ? probBg    : reboundBg;
  const fgFn    = isDown ? probFg    : reboundFg;
  const trigTxt = isDown ? "≥P" : "≤P";

  const th = (txt, extra='') =>
    `<th style="padding:5px 7px;border:1px solid #E8E8E4;background:#F5F5F2;
       color:#666;font-weight:500;white-space:nowrap;text-align:center;${{extra}}">${{txt}}</th>`;
  const td = (txt, bg='#fff', fg='#333', extra='') =>
    `<td style="padding:6px 5px;text-align:center;border:1px solid #E8E8E4;
       background:${{bg}};color:${{fg}};font-weight:600;${{extra}}">${{txt}}</td>`;

  let html = `<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;">`;

  for (const key of KEYS) {{
    const kd = probData[key];
    if (!kd) continue;
    const color = COLORS[key];
    const fwd   = kd.paired_days;

    html += `<div style="flex:1;min-width:260px;">
      <div style="font-size:12px;font-weight:600;color:${{color}};
                  padding-bottom:4px;border-bottom:2px solid ${{color}};margin-bottom:6px;">
        ${{key}}
        <span style="font-size:10px;font-weight:400;color:#AAA;">　觀察窗口：${{fwd}} 日</span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead><tr>
          ${{th('百分位','text-align:left;')}}
          ${{th('n')}}
          ${{labels.map((l,i) =>
            th(`${{l}}<br><span style="font-size:9px;font-weight:400;">${{fwd}}日內</span>`)
          ).join('')}}
        </tr></thead>
        <tbody>`;

    for (const t of kd.thresholds) {{
      const biasStr = (t.bias_val >= 0 ? '+' : '') + t.bias_val.toFixed(1) + '%';
      html += `<tr>
        ${{td(
          `<span style="font-weight:700;">${{trigTxt}}${{t.pct}}</span>
           <span style="font-size:10px;font-weight:400;color:#AAA;margin-left:4px;">${{biasStr}}</span>`,
          '#fff','#333','text-align:left;padding:5px 8px;'
        )}}
        ${{td(t.n_events,'#fff','#888')}}
        ${{moves.map(m => {{
          const p = t[`p${{m}}`] ?? null;
          return td(p !== null ? p.toFixed(0)+'%' : '—', bgFn(p), fgFn(p));
        }}).join('')}}
      </tr>`;
    }}

    html += `</tbody></table></div>`;
  }}

  html += `</div>`;

  el.innerHTML = html;
}}

// 回漲顏色（紅＝高機率偏多，台股慣例）
function reboundBg(p) {{
  if (p === null) return "#F5F5F2";
  if (p >= 70)   return "#FDECEA";
  if (p >= 50)   return "#FEF3E2";
  if (p >= 30)   return "#FEFCE2";
  return "#EAF5EA";
}}
function reboundFg(p) {{
  if (p === null) return "#BBB";
  if (p >= 70)   return "#C0392B";
  if (p >= 50)   return "#CA6F1E";
  if (p >= 30)   return "#9A7D0A";
  return "#1E8449";
}}

function renderPullbackTable(data) {{
  _buildProbTable('pullbackSection', data.pullback_prob, 'down');
}}

function renderReboundTable(data) {{
  _buildProbTable('reboundSection', data.rebound_prob, 'up');
}}

document.getElementById('stockInput').addEventListener('keydown', e => {{
  if (e.key === 'Enter') startAnalysis();
}});
</script>
</body>
</html>"""


# ══════════════════════════════════════════════
# 7. 伺服器啟動
# ══════════════════════════════════════════════
def _free_port(port: int) -> None:
    """若 port 被佔用，自動 kill 舊程序。"""
    import signal, subprocess
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True
        ).strip()
        if out:
            for pid in out.splitlines():
                os.kill(int(pid), signal.SIGKILL)
            time.sleep(0.3)
            print(f"   ⚠️  已清除佔用 port {port} 的舊程序")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass  # port 空的，或系統無 lsof（如 Replit）


def serve():
    _free_port(SERVER_PORT)
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", SERVER_PORT), _Handler)
    url    = f"http://localhost:{SERVER_PORT}"
    print(f"🚀 台股乖離率分析　→  {url}")
    print("   按 Ctrl+C 停止")
    if os.environ.get("PORT") is None:
        try:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    serve()
