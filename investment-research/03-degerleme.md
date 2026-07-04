# 03 — Değerleme

## Konunun Özeti
Değerleme, bir varlığın içsel değerini tahmin etme sanat ve bilimidir. Damodaran'ın
sınıflaması: (1) içsel değerleme (DCF), (2) göreli değerleme (çarpanlar),
(3) varlık bazlı değerleme, (4) opsiyon bazlı değerleme (reel opsiyonlar).
Hiçbir yöntem "doğru fiyatı" vermez; hepsi varsayım setinin çıktısıdır.
Amaç kesinlik değil, **fiyat ile değer arasındaki büyük kopuklukları** yakalamaktır.

## Temel Kavramlar ve Yöntemler

### DCF (İndirgenmiş Nakit Akışı)
- Değer = Σ [FCF_t / (1+r)^t] + Terminal Değer / (1+r)^n
- **FCFF** (firmaya serbest nakit akışı, WACC ile iskonto → şirket değeri) vs.
  **FCFE** (özkaynağa, cost of equity ile iskonto → özkaynak değeri).
- **WACC** = (E/V)·Re + (D/V)·Rd·(1−vergi). Re genellikle CAPM ile:
  Re = risksiz oran + β × hisse risk primi.
- **Terminal değer:** Gordon büyümesi TV = FCF_{n+1}/(r−g) veya çıkış çarpanı.
  Tipik DCF'te değerin %60–80'i terminal değerden gelir → model aslında
  "uzun vade varsayımları" modelidir.
- Duyarlılık: r ve g'de ±0,5 puan, değeri ±%15–30 oynatabilir. Tek nokta tahmini değil
  senaryo aralığı üretilmelidir.

### Göreli Değerleme (Çarpanlar)
- **P/E** = Fiyat / Hisse başı kâr. Basit ama: kâr negatifse çalışmaz, muhasebeye ve
  kaldıraca duyarlı, döngüsel şirketlerde zirvede ucuz görünür (klasik tuzak).
  Varyant: **Shiller CAPE** (10 yıllık enflasyon düzeltmeli kâr) — endeks düzeyinde
  uzun vadeli getiri ile ters ilişkili (Campbell & Shiller 1998), zamanlama aracı değil.
- **EV/EBITDA:** İşletme değeri (piyasa değeri + net borç + azınlık payı) / FAVÖK.
  Sermaye yapısından bağımsız karşılaştırma sağlar; capex'i yok sayması zaafı
  (EV/EBIT veya EV/FCF daha sıkı).
- **P/B** = Fiyat / Defter değeri. Bankalar/sigorta için hâlâ ana çarpan; maddi olmayan
  varlık yoğun şirketlerde (yazılım, marka) defter değeri anlamını yitirdi — Fama-French
  value faktörünün son dönem zayıflığının bir açıklaması. `[TARTIŞMALI]`
- **PEG** = P/E / büyüme oranı (Lynch'in popülerleştirdiği kaba kural: PEG < 1 cazip).
  Teorik temeli zayıf, hız pratik: büyümenin kalitesini ve süresini yok sayar.
- **Sektöre özgü:** EV/Satış (kârsız büyüme), EV/abone, $/rezerv varili,
  FFO/AFFO (REIT), P/NAV (holding, GYO).

### Comparable Analysis (Emsal Analizi)
- Benzer şirket seti kur → çarpanları hesapla → medyan/aralık uygula → farklılıklar
  için düzelt (büyüme, marj, risk).
- Zaafı: "emsal" seçimi sonucu belirler; tüm sektör balonsa emsaller de balonlu fiyat
  verir (1999 dot-com: "diğer internet şirketlerine göre ucuz").

### Sum of the Parts (SOTP)
- Farklı iş kollarını ayrı ayrı değerleyip toplama; holding ve konglomeralarda kullanılır.
- **Holding/konglomera iskontosu:** Piyasa, parçaların toplamına genellikle %10–30
  iskonto uygular (şeffaflık, sermaye tahsisi ve vergi nedenleriyle); Türkiye'de holding
  iskontoları tarihsel olarak daha derin olabilmiştir. `[EĞİTİM VERİSİ — oran aralıkları yaklaşıktır]`

### Varlık Bazlı Değerleme
- Tasfiye değeri, yenileme maliyeti, NAV. Graham'ın "net-net"i: piyasa değeri <
  (dönen varlıklar − tüm yükümlülükler) → istatistiksel olarak tarihte iyi çalıştı,
  bugün gelişmiş piyasalarda çok nadir.

## Nasıl Çalışır? (Pratik Akış)
1. İş modelini ve büyüme aşamasını belirle → yöntem seç (kârsız hızlı büyüyen:
   EV/Satış + uzun ufuklu DCF senaryoları; olgun nakit ineği: FCF bazlı DCF + EV/EBIT;
   banka: P/B–ROE ilişkisi; GYO/REIT: NAV, FFO).
2. Geliri aşağıdan yukarı (hacim × fiyat) projeksiyonla; marjları rekabet analiziyle bağla.
3. En az 3 senaryo (kötü/baz/iyi) + duyarlılık tablosu (r, g, marj).
4. Çapraz doğrulama: DCF sonucu ile emsal çarpanları karşılaştır; büyük uyumsuzluk
   varsa nedenini açıkla (senin varsayımın mı, piyasanın hatası mı?).
5. Güvenlik marjı uygula (tipik %20–40; belirsizlik arttıkça artar).

## Avantajlar
- DCF: değer sürücülerini (büyüme, marj, sermaye ihtiyacı, risk) açık hale getirir;
  disiplin sağlar.
- Çarpanlar: hızlı, piyasa fiyatlamasıyla bağlantılı, iletişimi kolay.
- SOTP: gizli değerleri (yan iş kolu, gayrimenkul, iştirak) ortaya çıkarır.

## Dezavantajlar
- DCF: "hassas görünümlü yanlışlık" riski — Excel'in kesinliği varsayımın belirsizliğini
  gizler. Terminal değer hâkimiyeti.
- Çarpanlar: göreli ucuzluk mutlak ucuzluk değildir; muhasebe farkları karşılaştırmayı bozar.
- Tüm yöntemler: değerleme yapanı değil varsayımı yansıtır — hedefe yürüyen "tersine
  mühendislik" değerlemesi yaygın meslek hastalığıdır (özellikle satış tarafı analistlerde).

## Riskler
- İskonto oranının yanlış seçimi (özellikle yüksek enflasyonlu TL bazlı değerlemede
  nominal/reel tutarsızlığı — nominal nakit akışı nominal oranla, reel reel ile).
- Büyümenin sermaye ihtiyacı olmadan geleceği varsayımı (büyüme bedavaya gelmez:
  g = yeniden yatırım oranı × ROIC).
- Döngü zirvesindeki marjı "normal" kabul etmek.
- Ülke riski/kur riskini iki kez saymak ya da hiç saymamak.

## Gerçek Örnekler
- **Dot-com (1999–2000):** "Göz küresi başına değer" gibi çarpan icatları; Cisco 2000'de
  ~200 milyar $ üzeri değerlemeden ~%80 düştü — kârlı ve gerçek bir şirket bile yanlış
  fiyatla kötü yatırımdır.
- **Nifty Fifty (1972):** "Her fiyata kalite" — Coca-Cola, Xerox vb. 40–80 P/E;
  1973–74'te %60–80 düşüş. Kaliteli şirket ≠ her fiyata alınır.
- **Damodaran vs. Tesla/NVIDIA:** Damodaran'ın halka açık DCF'leri, aynı şirket için
  farklı hikâyelerin nasıl 3–5 kat farklı değer ürettiğini belgeler — değerleme
  hikâyenin sayısallaşmasıdır. `[EĞİTİM VERİSİ]`

## Tarihsel Olaylar
- 1938: John Burr Williams, *The Theory of Investment Value* — DCF'in temeli.
- 1962: "Bond yield + growth" tarzı kaba modeller; 1960'lar CAPM'in doğuşu.
- 2000 ve 2021: düşük faizin uzun "duration"lı hisse değerlemelerini şişirmesi ve
  faiz şokuyla sönmesi (2022'de kârsız teknoloji endeksleri %60–80 düştü). `[EĞİTİM VERİSİ]`

## En Yaygın Hatalar
1. Terminal büyümeyi ekonominin nominal büyümesinin üzerine koymak (sonsuza dek
   ekonomiden hızlı büyüyen şirket ekonominin kendisi olurdu).
2. WACC'i her senaryoda sabit tutmak (kaldıraç/risk değişince değişir).
3. Çarpanı bağlamsız kullanmak: düşük P/E'nin nedeni çoğu zaman düşük kalite/büyümedir
   ("ucuzun ucuz olma nedeni").
4. Net borcu ve seyreltmeyi EV hesabında unutmak.
5. Tek senaryolu DCF sunmak.
6. Değer ile fiyat hedefini karıştırmak: değerleme "ne etmeli" sorusudur, "ne zaman
   oraya gider" sorusu değil.

## Uzman Görüşleri
- **Damodaran:** "Değerleme %80 hikâye %20 sayı değildir; ikisinin tutarlılığıdır."
  Her varlık değerlenebilir ama her değerleme aynı güvenilirlikte değildir.
- **Buffett:** Resmî DCF tablosu kullanmadığını söyler; "içsel değer" zihinsel modeldir,
  kesin hesap değil. "Yaklaşık haklı olmak, kesin yanılmaktan iyidir."
- **Graham:** Kesin değerlemeden çok güvenlik marjına ağırlık; "tam değeri bilmene gerek
  yok, birinin şişman olduğunu anlamak için kilosunu bilmen gerekmez".
- **McKinsey (Koller vd., *Valuation*):** Kurumsal standart; değer = ROIC ve büyümenin
  fonksiyonu, çarpanlar DCF'in kısaltmasıdır.
- **Çelişki:** Akademi CAPM/beta ile risk ölçümünü savunur; Buffett/Munger beta'yı
  "saçmalık" olarak reddeder (risk = kalıcı sermaye kaybı, oynaklık değil). İki çerçeve
  farklı sorulara cevap verir; portföy kuramı için beta, tekil işletme riski için
  nitel analiz daha uygundur.

## Akademik Çalışmalar
- Gordon (1962), temettü büyüme modeli.
- Campbell & Shiller (1988, 1998), CAPE ve uzun vadeli getiri tahmini.
- Fama & French (1992, 1993), değer priminin belgelenmesi (HML).
- Penman (2013), *Financial Statement Analysis and Security Valuation* — muhasebe
  temelli değerleme (residual income).
- Ohlson (1995), artık kâr (residual income) modeli, *Contemporary Accounting Research*.

## Kaynaklar
- Damodaran, *Investment Valuation*; *The Dark Side of Valuation*; ücretsiz veri:
  pages.stern.nyu.edu/~adamodar (hisse risk primleri, sektör çarpanları — düzenli güncellenir)
- Koller, Goedhart, Wessels (McKinsey), *Valuation*
- Mauboussin'in raporları (ROIC, beklentiler yatırımı — *Expectations Investing*)

## Güncel Gelişmeler
- 2026'da yükselen enflasyon/faiz ortamı, iskonto oranlarını ve hisse risk primini
  yukarı itiyor; uzun "duration"lı büyüme hisseleri değerleme açısından en duyarlı grup
  (bkz. dosya 13'teki güncel faiz verileri). `[DOĞRULANDI — faiz verisi; etki yorumu analitik çıkarımdır]`

## Sonuç
Değerleme bir aralık üretir, nokta değil. En sağlam pratik: (1) işe uygun yöntem,
(2) açık ve savunulabilir varsayımlar, (3) senaryo + duyarlılık, (4) çapraz doğrulama,
(5) güvenlik marjı. Model ne kadar sofistike olursa olsun, çıktı varsayım kalitesini aşamaz.

## Güven Seviyesi
- Yöntem tanımları ve matematik: **%95**
- Yöntem eleştirileri ve pratik akış: **%85**
- Tarihsel vaka rakamları: **%80** (yaklaşık değerler)
