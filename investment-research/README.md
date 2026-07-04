# Yatırım Araştırma Sistemi — Bilgi Tabanı

> **Amaç:** Yatırım dünyasını sıfırdan uzman seviyesine kadar kapsayan, kaynaklara
> dayalı, güven seviyesi işaretlenmiş sistematik bir bilgi tabanı oluşturmak.
>
> **Bu bir yatırım tavsiyesi DEĞİLDİR.** Buradaki hiçbir içerik alım/satım önerisi
> olarak yorumlanmamalıdır. Amaç eğitim ve araştırmadır.

## Dosya Dizini

| # | Dosya | Konu |
|---|-------|------|
| 01 | [01-makroekonomi.md](01-makroekonomi.md) | Faiz, enflasyon, resesyon, para/maliye politikası, merkez bankaları, tahvil/döviz piyasaları, likidite, sermaye akışları |
| 02 | [02-sirket-analizi.md](02-sirket-analizi.md) | Finansal tablolar, marjlar, ROE/ROIC/ROA, EBITDA/FAVÖK, net borç, içsel değer |
| 03 | [03-degerleme.md](03-degerleme.md) | DCF, çarpan analizi (P/E, EV/EBITDA, P/B, PEG), SOTP, varlık değerlemesi |
| 04 | [04-teknik-analiz.md](04-teknik-analiz.md) | Mum formasyonları, trend, RSI, MACD, Bollinger, EMA/SMA, VWAP, Fibonacci, hacim, market structure |
| 05 | [05-risk-yonetimi.md](05-risk-yonetimi.md) | Position sizing, stop-loss, Kelly, VaR, Sharpe/Sortino, max drawdown, çeşitlendirme |
| 06 | [06-portfoy-yonetimi.md](06-portfoy-yonetimi.md) | MPT, asset allocation, rebalancing, hedge, faktör yatırımı (value, momentum, quality…) |
| 07 | [07-varlik-siniflari.md](07-varlik-siniflari.md) | Hisse, ETF, tahvil, altın/gümüş, emtia, gayrimenkul/REIT, kripto, VC/PE |
| 08 | [08-turev-urunler.md](08-turev-urunler.md) | Opsiyonlar, vadeli işlemler, swap/forward, opsiyon stratejileri |
| 09 | [09-davranissal-finans.md](09-davranissal-finans.md) | Bilişsel önyargılar, sürü psikolojisi, FOMO, loss aversion, overconfidence |
| 10 | [10-algoritmik-islemler.md](10-algoritmik-islemler.md) | Quant, backtesting, walk-forward, Monte Carlo, ML/AI, istatistiksel arbitraj |
| 11 | [11-kuresel-piyasalar.md](11-kuresel-piyasalar.md) | ABD, Avrupa, Asya, Türkiye, gelişmekte olan ülkeler |
| 12 | [12-hukuk-ve-vergi.md](12-hukuk-ve-vergi.md) | Mevzuat, vergilendirme, regülasyonlar, risk bildirimleri |
| 13 | [13-guncel-durum-2026.md](13-guncel-durum-2026.md) | Temmuz 2026 itibarıyla doğrulanmış güncel piyasa/politika verileri (kaynaklı) |

## Metodoloji

### Araştırma Kuralları
1. Her bilgi için birincil kaynak aranır (akademik makale, resmî kurum, şirket raporu).
2. Kritik veriler en az iki bağımsız kaynakla doğrulanmaya çalışılır.
3. Her verinin **tarihi** belirtilir; eski veri güncelliğini yitirmiş olabilir.
4. Doğrulanamayan veya modelin eğitim verisine dayanan bilgiler açıkça işaretlenir.
5. Tahmin ile gerçekleşmiş veri asla karıştırılmaz.

### Kaynak Öncelik Sırası
1. Hakemli akademik makaleler (Journal of Finance, JFE, RFS…)
2. Resmî kurumlar (Fed, ECB, TCMB, BLS, TÜİK, IMF, BIS)
3. Düzenleyici otoriteler (SEC, ESMA, SPK, FINRA)
4. Şirket raporları (10-K, faaliyet raporları, hissedar mektupları)
5. Finansal veri sağlayıcıları (Bloomberg, Refinitiv, S&P, MSCI)
6. Üniversite yayınları (NBER, SSRN çalışma kâğıtları)
7. Saygın kitaplar (Graham, Marks, Damodaran, Bernstein…)
8. Uzman röportajları ve mektupları
9. Sosyal medya: **yalnızca destekleyici**, asla birincil kaynak değil.

### Güven Seviyesi Ölçeği
- **%90–100:** Akademik konsensüs veya resmî veri; birden çok kaynakla tutarlı.
- **%70–89:** Yaygın kabul gören, ancak yorum/uygulama farkları olan bilgi.
- **%50–69:** Uzmanlar arasında ciddi görüş ayrılığı olan veya bağlama bağlı bilgi.
- **<%50:** Spekülatif, doğrulanmamış veya güncelliği belirsiz — açıkça işaretlenir.

### Etiketler
- `[DOĞRULANDI — kaynak, tarih]` : Web/resmî kaynakla teyit edildi.
- `[EĞİTİM VERİSİ — ~Ocak 2026 kesiti]` : Model bilgisine dayanır, güncel teyit yok.
- `[TARTIŞMALI]` : Uzmanlar arasında çelişen görüşler var; iki taraf da sunulur.
- `[DOĞRULANMADI]` : Tek kaynak veya belirsiz kaynak; ihtiyatla kullanılmalı.

## Proje Mimarisi (Modüler Yapı)

```
investment-research/
├── README.md                  # Metodoloji, mimari, dizin (bu dosya)
├── PROJECT_STATUS.md          # Modül ilerleme durumu — kesinti sonrası buradan devam edilir
├── 01-makroekonomi.md         # Konu modülleri (her biri bağımsız, ayrı commit)
├── ...
├── 13-guncel-durum-2026.md    # Kaynaklı güncel veri anlık görüntüsü (her oturumda revize)
├── sablonlar/                 # Yeniden kullanılabilir analiz şablonları
│   └── sirket-analiz-sablonu.md
├── uzmanlar/                  # Yatırımcı strateji karşılaştırmaları (planlı)
├── kurum-raporlari/           # SEC/Fed/ECB/TCMB rapor sınıflandırması (planlı)
├── kitap-ozetleri/            # Yatırım kitabı analiz arşivi (planlı, artımlı büyür)
└── ajanlar/                   # Günlük piyasa takip ajanı tasarımı/kodu (planlı)
```

### Çalışma Modeli
1. **Modülerlik:** Her modül bağımsız tamamlanır ve **ayrı commit** ile kaydedilir.
   Tek oturumda projeyi bitirme hedefi yoktur; proje artımlı büyür.
2. **Kaldığı yerden devam:** Her modül sonunda `PROJECT_STATUS.md` güncellenir.
   Kesinti (API hatası, bağlantı, token sınırı) sonrası yeni oturum önce
   `PROJECT_STATUS.md`'yi okur; **tamamlanmış dosyalar yeniden oluşturulmaz**,
   sadece eksikler tamamlanır ve güncellenmesi gerekenler revize edilir.
3. **Sürüm kontrolü:** Tüm değişiklikler git geçmişinde izlenir; her bilginin
   "son güncelleme" tarihi dosya içinde ve commit geçmişinde görünür.
4. **Kaynak ve güven etiketi zorunlu:** Kaynaksız kesin ifade yazılmaz; emin
   olunmayan bilgi `[DOĞRULANMADI]` / `[EĞİTİM VERİSİ]` ile işaretlenir.

## Yaşayan Doküman İlkesi
Bu bilgi tabanı statik değildir. Yeni veri geldikçe ilgili dosya güncellenir,
`13-guncel-durum-2026.md` her araştırma oturumunda revize edilir ve önceki
analizlerle çelişen yeni bulgular açıkça not edilir.

---
*Son güncelleme: 2026-07-04*
