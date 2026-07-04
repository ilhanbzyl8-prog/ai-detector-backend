# 05 — Risk Yönetimi

## Konunun Özeti
Risk yönetimi, getiriden önce hayatta kalmayı güvence altına alma disiplinidir.
Temel aksiyom: **birleşik getiri geometriktir** — %50 kayıp, telafi için %100 kazanç
gerektirir; dolayısıyla büyük kayıplardan kaçınmak, büyük kazançlar kovalamaktan
matematiksel olarak daha değerlidir. Howard Marks: "Önce kaybetme; kazanç kendiliğinden gelir."

## Temel Kavramlar

### Position Sizing (Pozisyon Boyutlandırma)
- **Sabit yüzde riski:** İşlem başına sermayenin %0,5–2'si riske edilir.
  Boyut = (Sermaye × risk%) / (giriş − stop mesafesi).
- **Volatilite hedefleme:** Pozisyon, varlığın volatilitesiyle ters orantılı boyutlanır
  (ATR veya σ bazlı); portföy volatilitesini sabitler. CTA/risk-parite fonlarının standardı.
- Boyutlandırma, giriş sinyalinden daha belirleyicidir: aynı sinyal setiyle farklı
  boyutlandırma, iflasla zenginlik arasındaki farkı yaratabilir.

### Stop-Loss / Take-Profit
- **Stop-loss:** Önceden tanımlı çıkış; tezin geçersizleştiği fiyata konur (rastgele
  % değil). Türleri: sabit, iz süren (trailing), zaman stopu, volatilite (ATR) stopu.
- Ampirik durum: momentum/trend stratejilerinde stoplar drawdown'u azaltır
  (Kaminski & Lo 2014, "When Do Stop-Loss Rules Stop Losses?"); ortalamaya dönüş
  stratejilerinde ise getiriye zarar verebilir. **Stopun işe yarayıp yaramadığı
  stratejinin getiri dağılımına bağlıdır.** `[TARTIŞMALI — bağlama bağlı]`
- Uzun vadeli temel yatırımcılar (Buffett ekolü) fiyat stopu kullanmaz; onların
  "stopu" tezin bozulmasıdır. İki yaklaşım karıştırılmamalıdır.
- **Take-profit:** Hedefte kâr alma; R-multiple çerçevesi (risk birimi başına kazanç,
  ör. 1R risk / 2R hedef). Trend stratejileri sabit hedef yerine iz süren çıkış kullanır
  ("kârın koşmasına izin ver").

### Kelly Criterion
- f* = (bp − q) / b  (b: kazanç/kayıp oranı, p: kazanma olasılığı, q=1−p).
- Uzun vadeli sermaye büyüme hızını maksimize eder (Kelly 1956, Thorp uygulamaları).
- Pratik sorunlar: p ve b asla kesin bilinmez; tam Kelly aşırı oynaklık üretir
  (%50+ drawdown olası). Uygulayıcılar **yarım/çeyrek Kelly** kullanır.
- Thorp blackjack ve piyasada kullandı; Buffett/Munger'ın konsantre pozisyonları
  Kelly mantığıyla uyumludur (yüksek güvenli fırsata büyük bahis).

### Value at Risk (VaR)
- "X güven düzeyinde, Y sürede en fazla Z kaybederim" (ör. 1 gün %99 VaR = 1 M$).
- Yöntemler: parametrik (varyans-kovaryans), tarihsel simülasyon, Monte Carlo.
- **Kritik zaafı:** Kuyruğun ötesini söylemez; normal dağılım varsayımı şişman
  kuyrukları ıskalar. 2008'de bankaların VaR modelleri gerçekleşen kayıpların çok
  altında kaldı. Taleb'in ana eleştiri hedefi. Tamamlayıcısı: **Expected Shortfall
  (CVaR)** — kuyruğa düşünce ortalama kayıp; Basel III piyasa riskinde ES'e geçti.

### Performans/Risk Oranları
- **Sharpe** = (Getiri − risksiz) / σ. Standart ama simetrik: iyi oynaklığı da cezalandırır.
- **Sortino** = (Getiri − hedef) / aşağı yönlü sapma. Asimetrik stratejilerde daha adil.
- **Maximum Drawdown (MDD):** Zirveden dibe en büyük kayıp; yatırımcının psikolojik
  dayanıklılığıyla en doğrudan ilişkili ölçü. Calmar oranı = yıllık getiri / MDD.
- Dikkat: Sharpe geçmişe bakar ve kısa örneklemde yanıltır; negatif çarpıklıklı
  stratejiler (opsiyon satışı) krize kadar "mükemmel Sharpe" gösterir
  ("buharlı silindirin önünden kuruş toplamak").

### Çeşitlendirme
- Markowitz (1952): korelasyonu düşük varlıklar birlikte portföy riskini tekil
  risklerin ağırlıklı ortalamasının altına indirir — "finansta tek bedava öğle yemeği".
- Sınırı: krizde korelasyonlar 1'e yakınsar (2008'de neredeyse her şey birlikte düştü);
  çeşitlendirme normal zamanlarda çalışır, en çok ihtiyaç duyulduğunda zayıflar.
  Gerçek çeşitlendirme varlık sayısı değil, **getiri kaynağı** çeşitliliğidir.

## Nasıl Çalışır? (Bütünleşik Süreç)
1. **Hayatta kalma kısıtı:** Maksimum kabul edilebilir drawdown belirle (ör. %20).
2. İşlem başına risk bütçesi (%0,5–2) ve toplam korele risk limiti koy.
3. Her pozisyon: tez → geçersizlik noktası (stop) → boyut = risk bütçesi / stop mesafesi.
4. Portföy düzeyi: korelasyon, brüt/net maruziyet, senaryo/stres testi
   (tarihî şoklar: 2008, 2020, 2022 tekrarı simülasyonu).
5. Kayıt ve gözden geçirme: her işlem günlüğe; sistematik hata avı.

## Avantajlar
- İflas olasılığını (risk of ruin) yapısal olarak sıfıra yaklaştırır.
- Duygusal kararları kural setine devreder; panik anında karar verme ihtiyacını azaltır.
- Bileşik getirinin korunması: drawdown küçüldükçe toparlanma süresi hiperbolik kısalır.

## Dezavantajlar
- Getiriden feragat: sıkı risk kontrolü boğa piyasasında geride bıraktırır.
- Model riski: VaR/σ gibi ölçüler geçmiş rejime kalibredir; rejim değişince yanıltır.
- Aşırı stop kullanımı "testere" (whipsaw) maliyeti üretir.

## Riskler (Risk Yönetiminin Kendi Riskleri)
- **Sahte güven:** Modellenmiş risk ≠ gerçek risk (Knight belirsizliği modele girmez).
- **Kaldıraç yanılsaması:** Düşük volatilitede artırılan kaldıraç, volatilite
  sıçrayınca zorunlu satışa döner (volatility targeting'in prosiklik yan etkisi).
- **Likidite varsayımı:** Stop emri, boşluklu (gap) düşüşte istenen fiyattan dolmaz
  (2015 CHF şoku: stoplar %20+ ötede doldu).

## Gerçek Örnekler
- **LTCM (1998):** Nobelli ekip, 25:1+ kaldıraç, "birbirinden bağımsız" sanılan
  yakınsama işlemleri Rusya krizinde aynı anda çöktü; Fed koordineli kurtarma.
  Ders: korelasyon krizde içseldir; kaldıraç zaman tanımaz.
- **Amaranth (2006):** Tek trader'ın doğalgaz spread pozisyonu fonun %65'ini (~6 mlr $)
  bir ayda sildi. Ders: pozisyon konsantrasyon limitleri.
- **Archegos (2021):** Toplam getiri swap'larıyla gizlenmiş ~5–8x kaldıraç; iki günde
  ~20 mlr $ buharlaştı, Credit Suisse 5,5 mlr $ zarar. Ders: görünmeyen kaldıraç
  sistemik risktir.
- **Thorp (Princeton-Newport):** Kelly + arbitraj; ~20 yıl boyunca çeyrek asır
  neredeyse kayıpsız — pozitif örnek: boyutlandırma disiplininin kanıtı.

## Tarihsel Olaylar
- 1987: Portföy sigortası (dinamik hedge) çöküşü hızlandırdı — mekanik risk
  yönetiminin kendisi sistemik risk oldu.
- 1998 LTCM; 2007 "quant quake" (Ağustos, faktör kalabalıklaşması); 2008 VaR iflası;
- 2018 "Volmageddon" (XIV ürünü bir günde −%96); 2020 Mart marj sarmalı;
- 2022: 60/40'ın eşzamanlı düşüşü — tahvilin hedge rolünün enflasyon rejiminde
  çalışmadığının hatırlatıcısı.

## En Yaygın Hatalar
1. Riski getiriden sonra düşünmek (önce boyut, sonra hikâye olmalı).
2. Ortalama düşürmeyi (martingale) "indirimli alım" sanmak — planlı kademeli alım ile
   plansız zarar büyütme farklıdır.
3. Korelasyonu statik sanmak.
4. Volatiliteyi tek risk ölçüsü sanmak (asıl risk kalıcı sermaye kaybı).
5. Kaldıraçlı pozisyonda "uzun vadeli yatırımcıyım" demek (marj çağrısı vadeyi kısaltır).
6. Stopu sürekli geriye taşımak (riski büyütmenin en sinsi biçimi).
7. Kazanç sonrası risk artırma (house money etkisi, dosya 09).

## Uzman Görüşleri
- **Buffett:** "Kural 1: Para kaybetme. Kural 2: Kural 1'i unutma." + türevleri
  "finansal kitle imha silahı" (2002 mektubu — kendisi de seçici kullanır, tutarsızlık eleştirisi alır).
- **Taleb:** Kuyruk riski merkezde; "barbell" (aşırı güvenli + küçük agresif uçlar),
  VaR'a köklü itiraz. *Antifragile*: şoktan güçlenen yapılar kur.
- **Marks:** Risk = kayıp olasılığı; oynaklık değil. Risk en çok "risk yok" hissi
  yayıldığında birikir (*The Most Important Thing*).
- **Dalio:** Bilinmeyene karşı çeşitlendirme ("Holy Grail": 15–20 korelasyonsuz getiri
  akışı riski ~%80 düşürür).
- **Çelişki:** Taleb çeşitlendirmenin kuyrukta çöktüğünü, sadece dışbükey (convex)
  korumaların işe yaradığını savunur; Markowitz/Dalio geleneği çeşitlendirmeyi merkeze
  koyar. İkisi farklı risk türlerine bakar: olağan dalgalanma (çeşitlendirme çözer) vs.
  sistemik kuyruk (opsiyonellik/nakit çözer). İyi program ikisini birden kullanır.

## Akademik Çalışmalar
- Markowitz (1952), "Portfolio Selection", *Journal of Finance*.
- Kelly (1956), "A New Interpretation of Information Rate", *Bell System Technical Journal*.
- Artzner vd. (1999), tutarlı risk ölçüleri (VaR'ın alt-toplamsallık ihlali), *Mathematical Finance*.
- Kaminski & Lo (2014), stop-loss kurallarının rejime bağlı etkinliği, *Journal of Financial Markets*.
- Moreira & Muir (2017), "Volatility-Managed Portfolios", *Journal of Finance* —
  volatilite hedeflemenin Sharpe'ı iyileştirdiği bulgusu (sonraki literatür kısmen sorguladı). `[TARTIŞMALI]`

## Kaynaklar
- Thorp, *A Man for All Markets*; Poundstone, *Fortune's Formula* (Kelly tarihi)
- Taleb, *Fooled by Randomness*, *The Black Swan*, *Antifragile*
- Marks, *The Most Important Thing*; Lowenstein, *When Genius Failed* (LTCM)
- Basel Komitesi FRTB dokümanları (ES/VaR çerçevesi) — bis.org

## Güncel Gelişmeler
- 2026'nın jeopolitik enerji şoku ortamında kuyruk koruması (uzun volatilite,
  emtia maruziyeti) yeniden fiyatlanıyor; enflasyonist şoklarda tahvil hedge'inin
  sınırlılığı 2022'den sonra ikinci kez test ediliyor. `[EĞİTİM VERİSİ + dosya 13 makro verisi]`

## Sonuç
Risk yönetiminin hiyerarşisi: (1) iflastan kaçın (boyutlandırma, kaldıraç disiplini),
(2) drawdown'u yönet (stop/çeşitlendirme/nakit), (3) riski verimli harca (en iyi
fırsatlara risk bütçesi). Getiri tahmini yanılabilir; risk disiplini yanılmamalıdır —
uzun oyunda kalan kazanır.

## Güven Seviyesi
- Matematiksel kavramlar (Kelly, VaR, oranlar): **%95**
- Vaka çalışmaları: **%90**
- "En iyi pratik" önerileri: **%80** (stratejiye göre değişir)
- Stop-loss etkinliği: **%65 — bağlama bağlı, [TARTIŞMALI]**
