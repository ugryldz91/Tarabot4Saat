#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance taraması — PUBLIC endpoints (API key gerekmez)

Kapanan günlük mum için koşullar:
  1) Wilder RSI(14) < 30
  2) Close < Bollinger lower (20, 2σ)
  3) Son gün hacmi > önceki 3 gün ortalamasının 1.8×
İyileştirmeler (v4):
- BASE önceliği: data-api.binance.vision → api-gcp → data → api1/2/3/4
- 6 denemeye kadar exponential backoff + jitter, 429 (rate limit) algılama
- Timeout ↑ (45→70s), eşzamanlı istek sayısı ↓ (8→4)
- exchangeInfo başarısızsa ticker/price FALLBACK
- Mesajda taranan toplam coin sayısı
"""

import asyncio
import aiohttp
import os
import csv
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

DEFAULT_BASES = [
    "https://data-api.binance.vision",
    "https://api-gcp.binance.com",
    "https://data.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

SESSION_TIMEOUT = aiohttp.ClientTimeout(total=70)
UA_HEADERS = {"User-Agent": "UgurBinanceScan/1.0 (+github actions)"}

EXCLUDE_KEYWORDS = (
    "UP","DOWN","BULL","BEAR","2L","2S","3L","3S","4L","4S","5L","5S","PERP"
)

# ---------- Indicators ----------
def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.ewm(alpha=1/period, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-12)
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return lower, ma, upper

# ---------- HTTP helpers ----------
async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] = None) -> Any:
    base_delay = 0.8
    for attempt in range(6):
        try:
            async with session.get(url, params=params, headers=UA_HEADERS) as resp:
                if resp.status == 200:
                    return await resp.json()

                # Rate limit / geçici engel: bekleyip tekrar dene
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = base_delay
                    else:
                        sleep_s = base_delay * (2 ** attempt)
                    sleep_s += random.uniform(0, 0.6)
                    print(f"[warn] 429 rate limit, sleeping {sleep_s:.1f}s")
                    await asyncio.sleep(sleep_s)
                    continue

                txt = await resp.text()
                print(f"[warn] GET {url} -> {resp.status}, body={txt[:200]}")

        except asyncio.TimeoutError:
            print(f"[warn] timeout GET {url}")
        except Exception as e:
            print(f"[warn] error GET {url}: {e}")

        sleep_s = base_delay * (1.7 ** attempt) + random.uniform(0, 0.6)
        await asyncio.sleep(sleep_s)
    raise RuntimeError(f"GET {url} failed after retries.")

async def try_bases(path: str, params: Dict[str, Any] = None, bases: Optional[List[str]] = None) -> Any:
    bases = bases or DEFAULT_BASES
    async with aiohttp.ClientSession(timeout=SESSION_TIMEOUT) as session:
        last_err = None
        for base in bases:
            url = f"{base}{path}"
            try:
                return await fetch_json(session, url, params=params)
            except Exception as e:
                last_err = e
                print(f"[info] fallback -> {base} failed: {e}")
        if last_err:
            raise last_err
        raise RuntimeError("No base could be used.")

# ---------- Symbol sources ----------
async def get_spot_usdt_symbols_via_exchange_info(bases=None) -> List[str]:
    # SPOT ile daraltılmış exchangeInfo
    data = await try_bases("/api/v3/exchangeInfo", params={"permissions": "SPOT"}, bases=bases)
    out = []
    for s in data.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("isSpotTradingAllowed", False) and s.get("quoteAsset") == "USDT":
            sym = s.get("symbol","")
            if any(sym.endswith(k) for k in EXCLUDE_KEYWORDS):
                continue
            out.append(sym)
    return sorted(set(out))

async def get_spot_usdt_symbols_via_ticker(bases=None) -> List[str]:
    # exchangeInfo çalışmazsa ticker/price ile USDT paritelerini çıkar
    data = await try_bases("/api/v3/ticker/price", bases=bases)
    out = []
    for item in data:
        sym = item.get("symbol","")
        if sym.endswith("USDT") and not any(sym.endswith(k) for k in EXCLUDE_KEYWORDS):
            out.append(sym)
    return sorted(set(out))

async def get_spot_usdt_symbols(bases=None) -> List[str]:
    try:
        return await get_spot_usdt_symbols_via_exchange_info(bases=bases)
    except Exception as e:
        print(f"[info] exchangeInfo alınamadı, ticker fallback: {e}")
        return await get_spot_usdt_symbols_via_ticker(bases=bases)

# ---------- Klines ----------
async def get_klines(symbol: str, limit: int = 80, bases=None) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": "4h", "limit": limit}
    raw = await try_bases("/api/v3/klines", params=params, bases=bases)
    cols = ["openTime","open","high","low","close","volume","closeTime","quoteAssetVolume",
            "numberOfTrades","takerBuyBase","takerBuyQuote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    if df.empty:
        return df
    for c in ["open","high","low","close","volume","quoteAssetVolume","takerBuyBase","takerBuyQuote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["closeTime"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    return df

def check_conditions(df: pd.DataFrame) -> Tuple[bool, float]:
    if df is None or len(df) < 25:
        return (False, float("nan"))

    now_utc = datetime.now(timezone.utc)
    last_idx = len(df) - 1
    last_close_time = df["closeTime"].iloc[last_idx]
    # Son kapanmış 4s mumu
    i_cl = last_idx if now_utc >= last_close_time else last_idx - 1

    # RSI(14) ve BB(20) pencereleri için yeterli veri var mı?
    if i_cl < 20:  # hem BB(20) hem RSI(14) için güvenli alt sınır
        return (False, float("nan"))

    close = df["close"]
    volume = df["volume"]

    rsi = wilder_rsi(close, 14)
    bb_lower, _, _ = bollinger_bands(close, 20, 2.0)

    last_close = close.iloc[i_cl]
    last_rsi = rsi.iloc[i_cl]
    last_bb_lower = bb_lower.iloc[i_cl]

    # Hacim koşulu: son kapanmış mumun hacmini, önceki 3 kapanmış mumun ortalamasıyla kıyasla
    if i_cl < 3 or pd.isna(volume.iloc[i_cl]):
        return (False, float(last_rsi) if not pd.isna(last_rsi) else float("nan"))

    prev3 = volume.iloc[i_cl-3:i_cl]
    if prev3.isna().any():
        return (False, float(last_rsi) if not pd.isna(last_rsi) else float("nan"))

    vol_ok = volume.iloc[i_cl] > 1.8 * prev3.mean()
    cond_rsi = (not pd.isna(last_rsi)) and (last_rsi < 30.0)
    cond_bb  = (not pd.isna(last_bb_lower)) and (last_close < last_bb_lower)

    return (bool(cond_rsi and cond_bb and vol_ok),
            float(last_rsi) if not pd.isna(last_rsi) else float("nan"))

# ---------- Scan ----------
async def scan_all(bases=None) -> Tuple[List[Tuple[str,float]], int]:
    symbols = await get_spot_usdt_symbols(bases=bases)
    results: List[Tuple[str,float]] = []
    sem = asyncio.Semaphore(4)  # eşzamanlılık düşürüldü

    async def worker(sym: str):
        async with sem:
            try:
                df = await get_klines(sym, 80, bases=bases)
                ok, rsi_val = check_conditions(df)
                if ok:
                    results.append((sym, round(rsi_val, 2)))
            except Exception as e:
                print(f"[warn] {sym} klines hata: {e}")

    await asyncio.gather(*[asyncio.create_task(worker(s)) for s in symbols])
    results.sort(key=lambda x: x[1])
    return results, len(symbols)

# ---------- Telegram ----------
async def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram env eksik; mesaj gönderilmeyecek.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    async with aiohttp.ClientSession(timeout=SESSION_TIMEOUT, headers=UA_HEADERS) as s:
        try:
            async with s.post(url, json=payload) as resp:
                _ = await resp.text()
        except Exception as e:
            print(f"[warn] telegram send error: {e}")

def format_message(pairs: List[Tuple[str,float]], scanned: int) -> str:
    if not pairs:
        return f"(4s) Kriterlere uygun coin bulunamadı.\nTaranan toplam coin: {scanned}"
    lines = [f"(4s) Kriterlere uyan coinler (RSI) — Taranan toplam coin: {scanned}"]
    for sym, r in pairs:
        lines.append(f"- {sym}: RSI={r}")
    return "\n".join(lines)

def write_csv(pairs: List[Tuple[str,float]], out_dir=".") -> str:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    path = os.path.join(out_dir, f"scan_results_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["symbol","rsi"])
        for sym, rsi_val in pairs: w.writerow([sym, rsi_val])
    return path

# ---------- Main ----------
async def main():
    bases_env = os.getenv("BINANCE_BASES")
    bases = [b.strip() for b in bases_env.split(",")] if bases_env else None
    try:
        pairs, scanned = await scan_all(bases=bases)
    except Exception as e:
        msg = f"Binance API erişilemedi: {e}\nLütfen daha sonra tekrar deneyin veya farklı BASE URL deneyin."
        print(msg); await send_telegram(msg); return
    msg = format_message(pairs, scanned); print(msg)
    write_csv(pairs); await send_telegram(msg)

if __name__ == "__main__":
    asyncio.run(main())
