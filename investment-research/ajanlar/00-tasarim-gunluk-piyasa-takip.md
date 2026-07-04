# M18 Tasarım — Günlük Piyasa Takip Ajanı

> Durum: Tasarım v1.0 (2026-07-04). Kod: `piyasa-takip/` altında.
> Bu ajan **bilgi toplar ve raporlar; yatırım tavsiyesi üretmez.**

## 1. Amaç
Her gün (veya elle tetiklenince) temel piyasa göstergelerini ücretsiz kaynaklardan
çekip:
1. `ajanlar/raporlar/YYYY-MM-DD.md` altına tarihli, kaynaklı bir günlük rapor yazmak,
2. Eşik aşımı varsa (büyük hareket) raporun başına `⚠️ DİKKAT` bölümü eklemek,
3. Bilgi tabanının güncel durum dosyasının (`13-guncel-durum-2026.md`) elle
   revizyonu için hazır veri sağlamak (otomatik düzenleme YAPMAZ — insan onayı ilkesi).

## 2. Kapsam (v1 göstergeleri)
| Gösterge | Sembol | Kaynak | Not |
|---|---|---|---|
| S&P 500 | `^spx` | Stooq CSV | gecikmeli/kapanış verisi yeterli |
| Nasdaq 100 | `^ndx` | Stooq CSV | |
| BIST 100 | `^xu100` | Stooq CSV | sembol desteği ilk çalıştırmada doğrulanmalı `[DOĞRULANMADI]` |
| USD/TRY | `usdtry` | Stooq CSV | |
| EUR/USD | `eurusd` | Stooq CSV | |
| Altın (spot) | `xauusd` | Stooq CSV | |
| Brent | `cb.f` | Stooq CSV | vadeli sembolü; alternatifi doğrula `[DOĞRULANMADI]` |
| BTC, ETH | — | CoinGecko `simple/price` | anahtar gerektirmez, oran limiti ~10-30 istek/dk |
| Fed funds (efektif) | `DFF` | FRED `fredgraph.csv` | anahtarsız CSV indirme |
| ABD 10Y tahvil | `DGS10` | FRED `fredgraph.csv` | |

Kaynak seçim ilkesi: **anahtarsız + ücretsiz + CSV/JSON** → bağımlılık ve gizli anahtar
yönetimi yok. Doğruluk için ikinci kaynak çapraz kontrolü v2 hedefi (bkz. §8).

## 3. Mimari
```
piyasa-takip/
├── agent.py        # tek dosyalık ajan: çek → hesapla → raporla
├── config.py       # göstergeler, eşikler, dosya yolları (koddan ayrık)
├── test_agent.py   # ağ GEREKTİRMEYEN birim testleri (sahte veriyle)
└── raporlar/       # çıktı: YYYY-MM-DD.md + son durum: latest.json
```
- **Bağımlılık:** yalnızca Python 3.10+ standart kütüphanesi (`urllib`, `json`,
  `csv`, `datetime`). Bilinçli tercih: kurulumsuz taşınabilirlik.
- **Akış:** `fetch_all()` → her kaynak bağımsız try/except (birinin düşmesi raporu
  düşürmez; eksik veri raporda "VERİ YOK — kaynak hatası" satırı olur) →
  `latest.json` ile önceki güne göre % değişim → `build_report()` → dosyaya yaz.
- **Durum:** `latest.json` son başarılı değerleri ve zaman damgalarını tutar;
  % değişim ve "veri bayatlığı" (>3 gün eski) uyarısı buradan hesaplanır.

## 4. Eşik/Uyarı Kuralları (v1 — basit ve açıklanabilir)
| Koşul | Uyarı |
|---|---|
| Endeks günlük |Δ| ≥ %2 | ⚠️ büyük hareket |
| USD/TRY günlük |Δ| ≥ %1,5 | ⚠️ kur şoku adayı |
| BTC günlük |Δ| ≥ %5 | ⚠️ kripto oynaklığı |
| Altın veya Brent |Δ| ≥ %3 | ⚠️ emtia/jeopolitik sinyal |
| DGS10 günlük Δ ≥ 10 baz puan | ⚠️ tahvil hareketi |
Eşikler `config.py` içinde; gerekçe: sabit basit eşikler v1'de σ-bazlı dinamik
eşiklerden (v2) daha az veri gerektirir ve yanlış alarm analizi kolaydır.

## 5. Rapor Formatı (`raporlar/YYYY-MM-DD.md`)
```
# Günlük Piyasa Raporu — YYYY-MM-DD
⚠️ DİKKAT: (varsa eşik aşımı listesi)
| Gösterge | Değer | Günlük Δ% | Kaynak | Zaman |
...
Notlar: veri alınamayan kaynaklar, bayat veri uyarıları
Sorumluluk reddi (her raporda sabit)
```

## 6. Çalıştırma / Zamanlama
- Elle: `python3 agent.py` (repo kökünden veya klasöründen).
- Zamanlanmış (önerilen): GitHub Actions cron (`0 18 * * 1-5` UTC ≈ ABD kapanışı
  sonrası) → rapor dosyasını commit'leyen workflow. Workflow dosyası v1'e dahil
  edilmedi; kullanıcı onayıyla eklenecek (repoya CI eklemek görünür bir değişikliktir).
- **Ortam kısıtı notu:** Bu geliştirme sandbox'ında dış veri kaynakları ağ
  politikası gereği erişilemez (proxy 403, test edildi 2026-07-04). Bu yüzden canlı
  uçtan uca test kullanıcı ortamında yapılmalıdır; birim testleri ağsız çalışır.

## 7. Hata Yönetimi ve Dürüstlük İlkeleri
- Kaynak hatasında uydurma/tahmini değer YAZILMAZ; "VERİ YOK" yazılır.
- Her değerin yanında kaynak ve çekilme zamanı bulunur.
- Rapor yorum içermez; yorum insan işidir (bilgi tabanı kurallarıyla tutarlılık).
- Oran limitlerine saygı: istek başına tek çağrı, deneme arası bekleme, User-Agent başlığı.

## 8. Yol Haritası
- **v1 (bu sürüm):** çek → karşılaştır → raporla; birim testli; anahtarsız kaynaklar.
- **v2:** ikinci kaynakla çapraz doğrulama; σ-bazlı dinamik eşikler; TCMB EVDS ve
  TÜİK verisi (API anahtarı gerektirir — kullanıcı anahtarı sağlarsa); GitHub Actions.
- **v3:** haftalık özet + `13-guncel-durum` taslak güncellemesi üretme (yine insan onaylı).

## Güven Seviyesi
- Mimari/format kararları: tasarım tercihi (test edilecek).
- Stooq sembol listesi: **%70** — `^xu100` ve Brent sembolü ilk canlı çalıştırmada
  doğrulanmalı; hatalıysa `config.py`'de düzeltmek yeterli.
