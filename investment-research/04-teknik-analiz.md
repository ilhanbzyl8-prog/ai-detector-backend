# 04 — Teknik Analiz

## Konunun Özeti
Teknik analiz, geçmiş fiyat ve hacim verisinden gelecekteki fiyat hareketi hakkında
olasılıksal çıkarım yapma pratiğidir. Akademik statüsü tartışmalıdır: zayıf form etkin
piyasa hipotezi (EMH) çoğu klasik tekniğin işe yaramaması gerektiğini söyler; buna
karşın momentum ve trend takibi gibi bazı fiyat-temelli olgular akademik olarak da
belgelenmiştir. Dürüst çerçeve: **teknik analiz kesin tahmin aracı değil, risk yönetimi
ve zamanlama çerçevesidir.** `[TARTIŞMALI — bu dosyanın tamamı için geçerli üst not]`

## Temel Kavramlar

### Mum Formasyonları (Candlestick)
- Her mum: açılış, kapanış, yüksek, düşük. Gövde + fitiller.
- Yaygın formasyonlar: doji (kararsızlık), çekiç/hammer (dip dönüş adayı), yutan boğa
  (bullish engulfing), akşam yıldızı (evening star).
- Ampirik durum: tek başına mum formasyonlarının öngörü gücü üzerine kanıt zayıf ve
  karışıktır (Marshall vd. 2006, ABD hisselerinde ekonomik değer bulamadı). Bağlam
  (trend, seviye, hacim) olmadan kullanımı istatistiksel olarak savunulamaz. `[DOĞRULANMADI — lehte güçlü akademik kanıt yok]`

### Trend, Destek, Direnç
- **Trend:** Yükselen dipler/tepeler (yukarı) veya alçalan (aşağı). "Trend arkadaşındır"
  ilkesinin akademik akrabası zaman serisi momentumudur (aşağıda).
- **Destek/Direnç:** Alıcı/satıcı yoğunlaşması beklenen fiyat bölgeleri. Davranışsal
  temeli makul (çıpalama, bekleyen emirler, yuvarlak sayılar); kesin çizgi değil bölge
  olarak kullanılmalı.
- **Market Structure:** Tepe/dip dizilimi (HH-HL / LH-LL), kırılım (break of structure),
  likidite bölgeleri. Modern "price action" ekolünün dili; terminoloji yeni, kavramlar
  klasik Dow Teorisi'nin türevleridir.

### Göstergeler
- **SMA/EMA:** Basit/üssel hareketli ortalama. Trend filtresi olarak en test edilmiş
  kullanım: 200 günlük SMA üstü/altı rejim ayrımı — Meb Faber (2007) 10 aylık SMA
  kuralının getiriyi değil ama **drawdown'u** belirgin azalttığını gösterdi.
- **RSI (Wilder, 1978):** 0–100 momentum osilatörü; 70+ aşırı alım / 30− aşırı satım
  klasik okuma. Güçlü trendde RSI uzun süre "aşırı" kalabilir — ters işlem tuzağı.
- **MACD:** İki EMA farkı + sinyal çizgisi; momentum yön değişimi göstergesi.
- **Bollinger Bantları:** SMA ± 2 standart sapma; volatilite sıkışması (squeeze) ve
  aşırılık ölçümü. Banda değmek tek başına sinyal değildir (bant yürüyüşü olgusu).
- **VWAP:** Hacim ağırlıklı ortalama fiyat; kurumsal işlem kıyası ve gün içi referans.
  Kurumların icra (execution) kıyası olduğu için gün içinde gerçek davranışsal ağırlığı vardır.
- **Fibonacci düzeltmeleri (%38,2, %50, %61,8):** Matematiksel-mistik temeli zayıf;
  işe yaradığı ölçüde muhtemelen kendini gerçekleştiren kehanet + yaygın kullanımdır.
  `[DOĞRULANMADI — sağlam akademik destek yok]`
- **Hacim analizi:** Kırılımın hacimle teyidi, düşen hacimli rallilerin zayıflığı.
  Hacim-fiyat ilişkisi akademik olarak da bilgi taşır (Karpoff 1987 literatür özeti).

## Nasıl Çalışır? (Savunulabilir Kullanım Çerçevesi)
1. **Rejim tespiti:** Uzun MA'lara göre trend var mı? (Trend varken osilatör ters
   sinyalleri zayıftır; yatay piyasada trend sinyalleri zayıftır.)
2. **Seviye + tetik + teyit:** Bölge (destek/direnç) + fiyat davranışı (mum/kırılım)
   + hacim teyidi. Tek gösterge asla tek başına karar vermez.
3. **Risk tanımı önce:** Sinyal, giriş için değil **stop yerleşimi** için kullanılır;
   pozisyon boyutu stop mesafesinden türetilir (dosya 05).
4. **İstatistiksel doğrulama:** Kullanılan her kural backtest + walk-forward'dan
   geçmelidir (dosya 10); geçmeyen kural görsel hikâyedir.

## Avantajlar
- Net, kurallaştırılabilir giriş/çıkış disiplini sağlar (duyguyu azaltır).
- Risk yönetimiyle doğal entegrasyon (stop seviyeleri fiyat yapısından türetilir).
- Her varlıkta ve zaman ölçeğinde uygulanabilir; verisi ucuz ve erişilebilir.
- Momentum/trend bileşeni akademik olarak desteklenen tek fiyat-temelli kenardır.

## Dezavantajlar
- Çoğu klasik desenin ampirik doğrulaması zayıf; literatür yayın yanlılığıyla dolu.
- Aşırı serbestlik: aynı grafikte iki analist zıt sonuç çıkarabilir (yanlışlanamazlık).
- Görsel desen arama, insan beyninin gürültüde desen görme eğilimini (apophenia) sömürür.
- İşlem sıklığını artırır → maliyet ve vergi sürtünmesi (Barber & Odean 2000: aktif
  bireysel yatırımcılar piyasanın belirgin altında getiri elde etti).

## Riskler
- Kalabalık sinyallerde stop avlanması (bariz seviyelerin hemen ötesine dizilen stoplar).
- Geriye dönük uydurma (curve fitting) ile "mükemmel" görünen sistemlerin canlıda çökmesi.
- Kaldıraçla birleştiğinde küçük istatistiksel kenar bile iflas riskine dönüşebilir
  (dosya 05: risk of ruin).

## Gerçek Örnekler
- **Turtle Traders (1983):** Dennis & Eckhardt'ın kural bazlı trend takibi deneyi;
  öğretilebilir mekanik sistemle yıllarca yüksek getiri — trend takibinin ve
  disiplinin gücüne dair ünlü saha kanıtı (sonuçlar kişiler arasında çok değişkendi).
- **Managed futures / CTA endüstrisi:** Trend takip fonları (AQR, Man AHL vb.) 2008'de
  pozitif getiri sağladı ("kriz alfası"); 2010'ların sakin piyasasında zayıfladı,
  2022 enflasyon şokunda yeniden parladı (SG CTA endeksi 2022 ~+%20). `[EĞİTİM VERİSİ — yaklaşık]`
- **Karşı örnek:** Sayısız perakende "sinyal servisi"nin doğrulanabilir uzun vadeli
  getirisi yoktur; hayatta kalma yanlılığı sektörün pazarlamasını şişirir.

## Tarihsel Olaylar
- 1900'ler başı: Dow Teorisi (Charles Dow'un yazıları).
- 1978: Wilder, *New Concepts in Technical Trading Systems* (RSI, ATR, ADX).
- 1990'lar: Nison mum grafiklerini Batı'ya taşıdı.
- 1993: Jegadeesh & Titman momentumu akademik literatüre soktu — "teknik" bir olgunun
  akademide meşrulaşması.
- 2010'lar: HFT ve algoritmalar kısa vadeli desenlerin çoğunu arbitrajladı; perakende
  gün içi kenarları daraldı.

## En Yaygın Hatalar
1. Göstergeleri yığmak (hepsi aynı fiyattan türetilir → sahte teyit).
2. Backtest'siz kural kullanmak; ya da backtest'i aynı veride optimize edip inanmak.
3. Zaman ölçeği tutarsızlığı (günlük sinyalle girip 5 dakikalıkla panik çıkmak).
4. Aşırı alım/satımı otomatik "dönüş" sanmak.
5. Stop kullanmamak veya stopu sinyale göre değil acıya göre koymak.
6. Teknik analizi kesinlik aracı sanmak: en iyi ihtimalle küçük istatistiksel eğim verir.

## Uzman Görüşleri
- **Paul Tudor Jones:** 200 günlük ortalamanın altındaki hiçbir şeyi tutmam der;
  teknik + makro karışımı ile 1987 çöküşünü öngörüp kazandı.
- **Stanley Druckenmiller:** Pozisyon zamanlamasında grafikleri açıkça kullanır
  ("değerleme zamanlamayı söylemez, likidite ve teknikler söyler").
- **Eugene Fama (EMH):** Zayıf form etkinlikte fiyat geçmişinden kâr edilemez;
  klasik teknik analiz "astroloji" düzeyinde görülür.
- **Andrew Lo (MIT):** "Adaptif Piyasa Hipotezi" — kenarlar ekolojiktir; ortaya çıkar,
  sömürülür, kaybolur. Lo vd. (2000) bazı grafik desenlerinin istatistiksel bilgi
  içerdiğini buldu (ekonomik kârlılığı ayrı konu).
- **Çelişki net:** Akademi çekirdeği (Fama) vs. uygulayıcı efsaneleri (PTJ,
  Druckenmiller) doğrudan zıttır. Uzlaşı alanı: momentum/trend gibi sistematik,
  test edilebilir fiyat stratejileri; uzlaşmayan alan: öznel desen okuma.

## Akademik Çalışmalar
- Jegadeesh & Titman (1993), kesitsel momentum: 3–12 ay kazananlar kazanmaya devam
  eder, *Journal of Finance*.
- Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", *JFE*.
- Brock, Lakonishok & LeBaron (1992), MA ve kırılım kurallarında istatistiksel bilgi
  (sonraki çalışmalar veri madenciliği düzeltmesiyle zayıflattı — Sullivan vd. 1999).
- Lo, Mamaysky & Wang (2000), grafik desenlerinin biçimsel testi, *Journal of Finance*.
- Park & Irwin (2007), teknik analiz kârlılığı literatür taraması: sonuçlar karışık,
  yayın yanlılığı ciddi.
- Barber & Odean (2000), "Trading is Hazardous to Your Wealth", *Journal of Finance*.

## Kaynaklar
- Murphy, *Technical Analysis of the Financial Markets* (standart referans)
- Wilder (1978); Nison, *Japanese Candlestick Charting Techniques*
- Covel, *Trend Following*; Faber (2007), "A Quantitative Approach to Tactical Asset
  Allocation", *Journal of Wealth Management*
- Schwager, *Market Wizards* serisi (uygulayıcı röportajları — anekdot düzeyi)

## Güncel Gelişmeler
- Perakende akışının (0DTE opsiyonlar, sosyal medya koordinasyonu) kısa vadeli fiyat
  yapısını değiştirmesi; klasik desenlerin istatistiği rejimden rejime kayıyor. `[EĞİTİM VERİSİ]`
- ML tabanlı desen tanıma klasik göstergelerin yerini alıyor (dosya 10).

## Sonuç
Teknik analizden savunulabilir biçimde geriye kalan: (1) trend/momentum rejim filtreleri
(akademik destekli), (2) fiyat yapısından türetilmiş risk noktaları (stop disiplini),
(3) hacim/likidite okuması. Öznel desen okuma ise test edilmedikçe hikâyedir. Teknik
analiz tek başına kenar değildir; risk yönetimiyle birleşince işlem çerçevesi olur.

## Güven Seviyesi
- Gösterge tanımları: **%95**
- Momentum/trend etkisinin varlığı: **%85** (güçlü literatür; gelecekte devamı garanti değil)
- Klasik desenlerin (mum, Fibonacci) öngörü gücü: **%35 — DOĞRULANMADI**
- Uygulayıcı anekdotları: **%60** (doğrulanamaz performans iddiaları içerir)
