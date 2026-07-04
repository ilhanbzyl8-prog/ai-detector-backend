# 10 — Algoritmik ve Kantitatif İşlemler

## Konunun Özeti
Kantitatif yatırım, yatırım kararlarını test edilebilir kurallara ve istatistiksel
modellere dayandırır. Yelpaze: basit kural bazlı sistemlerden (aylık momentum) mikro
saniyelik HFT'ye ve ML tabanlı tahmin modellerine uzanır. Alanın çekirdek sorunu
teknik değil epistemolojiktir: **finansal veri az, gürültülü ve rejim-değişkendir;
aşırı uyum (overfitting) her yerdedir.** Başarılı quant pratiği, model kurma değil
kendini kandırmama disiplinidir.

## Temel Kavramlar

### Quant Strateji Türleri
- **İstatistiksel arbitraj:** Eşleştirilmiş işlem (pairs trading — Gatev vd. 2006),
  kointegrasyon bazlı sepetler, ortalamaya dönüş. Kenar küçük, kaldıraç ve hız gerekir;
  kalabalıklaşmaya duyarlı (Ağustos 2007 quant quake).
- **Faktör/stil sistematiği:** Value, momentum, quality, carry — düşük frekans,
  kapasiteli (dosya 06).
- **Trend takibi (CTA):** Zaman serisi momentumu, çok varlıklı vadeli işlemler.
- **HFT/piyasa yapıcılık:** Spread ve mikro yapı kenarları; teknoloji yarışı,
  perakendeye kapalı alan.
- **ML tabanlı tahmin:** Kesitsel getiri tahmini, NLP ile haber/duyarlılık, alternatif
  veri (uydu görüntüsü, kredi kartı akışı).

### Backtesting
- Stratejinin tarihsel simülasyonu. Geçerlilik koşulları:
  - **Look-ahead bias yok:** Karar anında bilinmeyen veri kullanılmaz (ör. bilanço
    verisi açıklanma tarihiyle, revizyonsuz haliyle — point-in-time veri).
  - **Survivorship bias yok:** Batık/delist şirketler veri setinde kalmalı.
  - **Gerçekçi maliyet:** Komisyon + spread + piyasa etkisi (büyük emirlerde belirleyici).
  - **Örneklem dışı test:** Veri, geliştirme/doğrulama/test olarak ayrılır.
- **Çoklu test problemi:** 1000 strateji dene, en iyisini raporla → "keşif" garantidir.
  Harvey & Liu: t-istatistiği eşiği çoklu test için 3,0+ olmalı (klasik 2,0 değil).
  Bailey & López de Prado: "deflated Sharpe ratio", deneme sayısına göre düzeltme.

### Walk-Forward Analysis
- Parametreleri geçmiş pencerede optimize et → takip eden pencerede test et → pencereyi
  kaydır. Overfitting'e karşı temel savunma; parametre istikrarsızlığı (her pencerede
  bambaşka optimum) stratejinin kırılganlık işaretidir.

### Monte Carlo
- Kullanımları: (1) işlem sırası permütasyonu ile drawdown dağılımı tahmini,
  (2) getiri örnekleme ile sonuç yelpazesi, (3) emeklilik/plan simülasyonu.
- Zaafı: girdiler tarihsel dağılımsa kuyruklar eksik kalır; blok bootstrap ve şişman
  kuyruklu dağılımlar kısmi çözüm.

### Machine Learning / AI
- Uygulama alanları: kesitsel hisse seçimi (gradient boosting, nöral ağlar — Gu, Kelly
  & Xiu 2020: ML yöntemleri klasik doğrusal modelleri örneklem dışında yendi),
  NLP duyarlılık (kazanç çağrısı tonu), rejim sınıflama, icra optimizasyonu.
- Finansal ML'in özel zorlukları (López de Prado, *Advances in Financial ML*):
  düşük sinyal/gürültü, örtüşen örnekler, rejim kayması, geriye dönük etiketleme.
  Önerileri: purged cross-validation, kombinatoryal CV, meta-etiketleme.
- LLM'ler (2023–): haber/rapor özetleme ve sinyal çıkarımı; kanıt: kısa vadeli tahmin
  gücü var ama hızla arbitrajlanıyor. `[EĞİTİM VERİSİ — hızlı gelişen alan]`

## Nasıl Çalışır? (Sağlam Araştırma Süreci)
1. **Ekonomik hipotez önce:** "Neden bu kenar var ve neden bana ödeniyor?" (risk primi mi,
   davranışsal mı, yapısal mı?) Cevapsızsa veri madenciliğidir.
2. Point-in-time, hayatta kalma yanlılığı olmayan veri kur.
3. Basit modelle başla; her karmaşıklık katmanının örneklem dışı katkısını kanıtla.
4. Maliyet/kapasite analizi: kâğıt üzerinde kârlı ≠ uygulanabilir.
5. Walk-forward + deflated Sharpe + parametre duyarlılığı.
6. Küçük sermayeyle canlı doğrulama (paper→pilot→ölçek); canlı/backtest sapması izlenir.
7. Bozulma protokolü: hangi metrik ne kadar bozulursa strateji emekli edilir (baştan yazılır).

## Avantajlar
- Duygusuz, tekrarlanabilir, ölçeklenebilir icra; 7/24 izleme.
- Test edilebilirlik: fikir kanıta bağlanır.
- Çeşitlendirme: onlarca küçük kenar tek portföyde birleşir (Renaissance modeli).

## Dezavantajlar
- Overfitting endüstriyel ölçekte kendini kandırma üretir; yayınlanan anomalilerin
  çoğu yayın sonrası zayıflar (McLean & Pontiff 2016: ortalama ~%50+ bozulma).
- Rejim değişimi: geçmişe kalibre model, yapısal kırılmada körleşir.
- Teknoloji/veri maliyeti; HFT'de silahlanma yarışı.
- "Kara kutu" riski: modelin neden çalıştığı bilinmezse ne zaman bozulduğu da bilinmez.

## Riskler
- **Kalabalıklaşma:** Aynı sinyale yığılan fonların eşzamanlı çözülmesi
  (Ağu 2007: istatistiksel arbitraj fonları günler içinde %20–30 kaybetti, Khandani & Lo 2011).
- **Teknik arıza:** Knight Capital (2012) — konuşlandırma hatası 45 dakikada 440 M$
  kaybettirdi, şirket bitti.
- **Flash crash (6 Mayıs 2010):** Algoritmik likidite çekilmesi; Dow dakikalar içinde
  ~%9 düştü. Piyasa mikro yapısının kırılganlığı.
- **Model riski + kaldıraç:** LTCM prototip vaka (dosya 05).

## Gerçek Örnekler
- **Renaissance Medallion:** 1988–2018 ortalama brüt ~%66/yıl (Zuckerman, *The Man Who
  Solved the Market*) — kapalı fon, dış sermaye yok; quant'ın tavanı ama tekrarlanamaz
  istisna. Aynı şirketin halka açık fonları (RIEF) çok daha sıradan performans gösterdi —
  kapasite ve frekans farkının kanıtı.
- **Two Sigma, DE Shaw, Citadel:** Çok stratejili quant platformların kalıcı başarısı.
- **Zillow Offers (2021):** ML fiyat modeliyle ev alım-satımı; model + icra + seçim
  yanlılığı (adverse selection) → ~500 M$+ zarar, iş kapatıldı. ML'in gerçek dünya
  seçim yanlılığına çarpması vakası.
- **Perakende algo-trading:** Sağ kalım yanlılığıyla pazarlanan kurs/bot endüstrisi;
  bağımsız doğrulanmış perakende algo başarı verisi yok denecek kadar azdır. `[DOĞRULANMADI]`

## Tarihsel Olaylar
- 1970'ler: Thorp'un warrant arbitrajı (ilk sistematik hedge fon sayılır).
- 1980'ler: Morgan Stanley APT grubu → pairs trading'in doğuşu; 1988 Medallion.
- 1998 LTCM; 2007 quant quake; 2010 flash crash; 2012 Knight; 2015 ETF açılış
  kopukluğu (24 Ağustos); 2018/2020 volatilite hedefleme sarmalları.

## En Yaygın Hatalar
1. Aynı veride optimize edip aynı veride "test" etmek.
2. Maliyetsiz backtest (özellikle yüksek rotasyonlu stratejilerde ölümcül).
3. Parametre sayısını veri miktarına göre sınırsız artırmak.
4. Sharpe'a tek başına güvenmek (çarpıklık/kuyruk gizler — kısa vol stratejileri).
5. Canlı sapmayı "geçici" diye izlemeye devam etmek (bozulma protokolü yokluğu).
6. Başkasının backtest'ine (kurs, Twitter, satıcı) itibar etmek.
7. Kaldıracı backtest MDD'sine göre ayarlamak (gerçek MDD her zaman daha derindir).

## Uzman Görüşleri
- **Jim Simons:** "Modele sadık kal; insan müdahalesi modeli bozar." Sır: veri kalitesi,
  kadro ve icra — tek sihirli sinyal değil.
- **López de Prado:** Finansal ML'de standart CV geçersizdir; süreç (purging,
  embargo, deflated metrics) her şeydir.
- **Asness:** Sistematik faktörler + disiplin; "quant kışlarında" (2018–20) bile terk etme.
- **Taleb (eleştirel):** Geçmiş veriden kuyruk riski öğrenilemez; optimizasyon kırılganlık üretir.
- **Thorp:** Kenarın kaynağını matematiksel olarak bilmeden bahis büyütme (Kelly bağlantısı, dosya 05).
- **Çelişki:** "Veri konuşsun" (pür ML) vs. "önce ekonomik hipotez" (Asness, AQR ekolü).
  Medallion pür örüntü tarafında başarılı tek büyük örnek; replikasyonu yok → temkinli
  varsayılan: hipotez-öncelikli yaklaşım.

## Akademik Çalışmalar
- Gatev, Goetzmann & Rouwenhorst (2006), pairs trading, *RFS*.
- Khandani & Lo (2011), Ağustos 2007 quant quake analizi, *Journal of Financial Markets*.
- Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns" (factor zoo), *RFS*.
- McLean & Pontiff (2016), yayın sonrası anomali bozulması, *JF*.
- Bailey & López de Prado (2014), deflated Sharpe ratio, *Journal of Portfolio Management*.
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", *RFS*.
- Kirilenko vd. (2017), flash crash mikro yapı analizi, *JF*.

## Kaynaklar
- López de Prado, *Advances in Financial Machine Learning*
- Chan, *Quantitative Trading* ve *Algorithmic Trading* (giriş düzeyi, dürüst)
- Zuckerman, *The Man Who Solved the Market*; Patterson, *The Quants*
- Açık araçlar: backtrader/zipline/vectorbt (Python), QuantConnect; veri: CRSP/Compustat
  (akademik standart), point-in-time ticari setler

## Güncel Gelişmeler
- LLM ajanlarının araştırma hattına girişi (fikir üretimi, kod, rapor taraması);
  sinyal yarı-ömürlerinin kısalması. `[EĞİTİM VERİSİ]`
- Perakende algo platformlarının yaygınlaşması regülatör radarında. `[EĞİTİM VERİSİ]`

## Sonuç
Quant'ta başarının sırası: veri kalitesi > süreç disiplini (anti-overfitting) >
maliyet gerçekçiliği > model karmaşıklığı. Parlak model en son gelir. Bireysel
uygulayıcı için gerçekçi hedef HFT değil; düşük frekanslı, ekonomik hipotezli,
mütevazı Sharpe'lı sistemlerdir — ve her backtest, aksi kanıtlanana kadar suçludur.

## Güven Seviyesi
- Metodoloji (backtest hijyeni, walk-forward, çoklu test): **%90**
- Vaka çalışmaları: **%85**
- ML'in kalıcı alfa ürettiği iddiası: **%55 — [TARTIŞMALI]**
- Endüstri performans anlatıları: **%70** (kapalı veri, doğrulama sınırlı)
