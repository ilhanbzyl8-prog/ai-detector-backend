"""Günlük piyasa takip ajanı (v1).

Kullanım:
    python3 agent.py            # bugünün raporunu üretir
    python3 agent.py --dry-run  # dosya yazmadan raporu stdout'a basar

Davranış sözleşmesi (tasarım §7):
- Kaynak hatasında değer UYDURULMAZ; gösterge raporda "VERİ YOK" olarak görünür.
- Her değerin yanında kaynak ve çekilme zamanı yazılır.
- Rapor yorum içermez; eşik aşımı yalnızca mekanik "DİKKAT" satırı üretir.

Yalnızca standart kütüphane kullanılır (tasarım §3).
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import config


@dataclass
class Reading:
    """Tek göstergenin tek günlük okuması."""

    key: str            # kaynak içi sembol/seri kimliği
    name: str           # raporda görünen ad
    value: float | None  # None => VERİ YOK
    asof: str           # kaynağın bildirdiği veri tarihi (YYYY-MM-DD) veya ""
    source: str         # "stooq" | "coingecko" | "fred"
    threshold: float    # uyarı eşiği (% veya baz puan — is_rate belirler)
    is_rate: bool = False  # True: değişim baz puan (mutlak fark×100) olarak ölçülür
    error: str = ""     # hata mesajı (value None ise nedeni)


def _http_get(url: str) -> str:
    """Basit GET; config'teki deneme sayısı kadar tekrar dener."""
    last_err: Exception | None = None
    for attempt in range(config.HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
            with urllib.request.urlopen(req, timeout=config.HTTP_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # ağ hataları gösterge bazında tolere edilir
            last_err = exc
            if attempt < config.HTTP_RETRIES:
                time.sleep(config.RETRY_WAIT)
    raise RuntimeError(f"{url} alınamadı: {last_err}")


# ---------------------------------------------------------------- kaynaklar

def parse_stooq_csv(text: str, key: str, name: str, threshold: float) -> Reading:
    """Stooq anlık CSV'sini çözer. Beklenen başlık: Symbol,Date,Time,...,Close,..."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return Reading(key, name, None, "", "stooq", threshold, error="boş CSV")
    row = rows[0]
    close = row.get("Close", "")
    if close in ("", "N/D", "N/A"):  # Stooq bilinmeyen sembolde N/D döner
        return Reading(key, name, None, "", "stooq", threshold,
                       error=f"kapanış yok (sembol geçersiz olabilir: {close!r})")
    try:
        value = float(close)
    except ValueError:
        return Reading(key, name, None, "", "stooq", threshold,
                       error=f"sayı çözülemedi: {close!r}")
    return Reading(key, name, value, row.get("Date", ""), "stooq", threshold)


def fetch_stooq() -> list[Reading]:
    out: list[Reading] = []
    for symbol, (name, threshold) in config.STOOQ_SYMBOLS.items():
        url = config.STOOQ_URL.format(symbol=urllib.request.quote(symbol))
        try:
            out.append(parse_stooq_csv(_http_get(url), symbol, name, threshold))
        except Exception as exc:
            out.append(Reading(symbol, name, None, "", "stooq", threshold, error=str(exc)))
    return out


def parse_coingecko_json(text: str) -> list[Reading]:
    data = json.loads(text)
    out = []
    for cid, (name, threshold) in config.COINGECKO_IDS.items():
        usd = data.get(cid, {}).get("usd")
        if usd is None:
            out.append(Reading(cid, name, None, "", "coingecko", threshold,
                               error="yanıt alanı eksik"))
        else:
            out.append(Reading(cid, name, float(usd),
                               datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                               "coingecko", threshold))
    return out


def fetch_coingecko() -> list[Reading]:
    ids = ",".join(config.COINGECKO_IDS)
    try:
        return parse_coingecko_json(_http_get(config.COINGECKO_URL.format(ids=ids)))
    except Exception as exc:
        return [Reading(cid, name, None, "", "coingecko", th, error=str(exc))
                for cid, (name, th) in config.COINGECKO_IDS.items()]


def parse_fred_csv(text: str, series: str, name: str, threshold: float) -> Reading:
    """fredgraph.csv: 'DATE,<SERIES>' başlıklı; '.' değeri eksik gözlem demektir."""
    rows = list(csv.reader(io.StringIO(text)))
    for date, raw in reversed(rows[1:]):  # sondan başa ilk geçerli gözlem
        if raw not in (".", ""):
            try:
                return Reading(series, name, float(raw), date, "fred",
                               threshold, is_rate=True)
            except ValueError:
                break
    return Reading(series, name, None, "", "fred", threshold, is_rate=True,
                   error="geçerli gözlem bulunamadı")


def fetch_fred() -> list[Reading]:
    out: list[Reading] = []
    for series, (name, threshold) in config.FRED_SERIES.items():
        url = config.FRED_URL.format(series=series)
        try:
            out.append(parse_fred_csv(_http_get(url), series, name, threshold))
        except Exception as exc:
            out.append(Reading(series, name, None, "", "fred", threshold,
                               is_rate=True, error=str(exc)))
    return out


def fetch_all() -> list[Reading]:
    return fetch_stooq() + fetch_coingecko() + fetch_fred()


# ------------------------------------------------------- durum ve hesaplama

def load_state() -> dict:
    if config.STATE_FILE.exists():
        return json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(readings: list[Reading]) -> None:
    """Yalnızca başarılı okumaları yazar; başarısız gösterge eski değerini korur."""
    state = load_state()
    for r in readings:
        if r.value is not None:
            state[r.key] = asdict(r)
    config.RAPOR_DIR.mkdir(parents=True, exist_ok=True)
    config.STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_change(r: Reading, state: dict) -> tuple[float | None, bool]:
    """(değişim, eşik_aşıldı_mı). Faizlerde baz puan, diğerlerinde % değişim.

    Önceki değer yoksa veya aynı 'asof' gününe aitse (yeni gözlem yok) None döner.
    """
    prev = state.get(r.key)
    if r.value is None or not prev or prev.get("value") in (None, 0):
        return None, False
    if prev.get("asof") == r.asof:
        return None, False  # aynı gözlem; değişim hesabı anlamsız
    if r.is_rate:
        delta = (r.value - prev["value"]) * 100.0  # yüzde puan -> baz puan
    else:
        delta = (r.value - prev["value"]) / prev["value"] * 100.0
    return delta, abs(delta) >= r.threshold


def is_stale(r: Reading, today: datetime) -> bool:
    if r.value is None or not r.asof:
        return False
    try:
        asof = datetime.strptime(r.asof, "%Y-%m-%d")
    except ValueError:
        return False
    return (today.replace(tzinfo=None) - asof).days > config.STALE_DAYS


# ----------------------------------------------------------------- raporlama

def build_report(readings: list[Reading], state: dict,
                 now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    lines = [f"# Günlük Piyasa Raporu — {today}", ""]

    alerts, notes = [], []
    rows = []
    for r in readings:
        delta, breached = compute_change(r, state)
        unit = "bp" if r.is_rate else "%"
        if r.value is None:
            rows.append((r.name, "VERİ YOK", "—", r.source, r.asof or "—"))
            notes.append(f"- {r.name}: veri alınamadı ({r.error})")
            continue
        delta_str = f"{delta:+.2f}{unit}" if delta is not None else "—"
        rows.append((r.name, f"{r.value:,.2f}", delta_str, r.source, r.asof or "—"))
        if breached:
            alerts.append(f"- **{r.name}**: {delta_str} (eşik: {r.threshold}{unit})")
        if is_stale(r, now):
            notes.append(f"- {r.name}: veri {r.asof} tarihli — BAYAT olabilir")

    if alerts:
        lines += ["## ⚠️ DİKKAT — eşik aşımı", *alerts, ""]

    lines += ["| Gösterge | Değer | Günlük Δ | Kaynak | Veri tarihi |",
              "|---|---|---|---|---|"]
    lines += [f"| {n} | {v} | {d} | {s} | {a} |" for n, v, d, s, a in rows]

    if notes:
        lines += ["", "## Notlar", *notes]

    lines += ["", "---", f"*Üretim zamanı: {now.strftime('%Y-%m-%d %H:%M UTC')}. "
              f"{config.DISCLAIMER}*", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    readings = fetch_all()
    state = load_state()
    report = build_report(readings, state)
    if dry_run:
        print(report)
        return 0
    config.RAPOR_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RAPOR_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    out.write_text(report, encoding="utf-8")
    save_state(readings)  # durumu rapor yazıldıktan sonra güncelle
    ok = sum(1 for r in readings if r.value is not None)
    print(f"Rapor yazıldı: {out} ({ok}/{len(readings)} gösterge başarılı)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
