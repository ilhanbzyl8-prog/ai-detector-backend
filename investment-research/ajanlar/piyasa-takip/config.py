"""Günlük piyasa takip ajanı yapılandırması.

Göstergeler, eşikler ve dosya yolları koddan ayrık tutulur; sembol düzeltmesi
veya eşik ayarı için agent.py'ye dokunmak gerekmez. Tasarım: ../00-tasarim-*.md
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAPOR_DIR = BASE_DIR / "raporlar"
STATE_FILE = RAPOR_DIR / "latest.json"

HTTP_TIMEOUT = 20  # saniye
HTTP_RETRIES = 2
RETRY_WAIT = 3  # saniye
USER_AGENT = "piyasa-takip-ajani/1.0 (kisisel arastirma; dusuk frekans)"

# Veri bu kadar günden eskiyse raporda "bayat veri" uyarısı verilir.
STALE_DAYS = 3

# Stooq anlık CSV uç noktası: sembol -> (görünen ad, uyarı eşiği %)
# Not: ^xu100 ve cb.f sembolleri ilk canlı çalıştırmada doğrulanmalı (tasarım §2).
STOOQ_SYMBOLS = {
    "^spx": ("S&P 500", 2.0),
    "^ndx": ("Nasdaq 100", 2.0),
    "^xu100": ("BIST 100", 2.0),
    "usdtry": ("USD/TRY", 1.5),
    "eurusd": ("EUR/USD", 1.5),
    "xauusd": ("Altın (spot, USD)", 3.0),
    "cb.f": ("Brent petrol", 3.0),
}
STOOQ_URL = "https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"

# CoinGecko: kimliksiz basit fiyat ucu. id -> (görünen ad, uyarı eşiği %)
COINGECKO_IDS = {
    "bitcoin": ("Bitcoin (USD)", 5.0),
    "ethereum": ("Ethereum (USD)", 5.0),
}
COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
)

# FRED anahtarsız CSV indirme: seri -> (görünen ad, uyarı eşiği)
# Faiz serilerinde eşik yüzde değişim değil, mutlak baz puan farkıdır.
FRED_SERIES = {
    "DFF": ("Fed funds (efektif, %)", 10.0),   # baz puan
    "DGS10": ("ABD 10Y tahvil (%)", 10.0),     # baz puan
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

DISCLAIMER = (
    "Bu rapor otomatik üretilmiştir; yalnızca bilgilendirme amaçlıdır ve "
    "yatırım tavsiyesi değildir. Veriler gecikmeli/ücretsiz kaynaklardandır; "
    "işlem kararı için doğrulanmış birincil kaynak kullanınız."
)
