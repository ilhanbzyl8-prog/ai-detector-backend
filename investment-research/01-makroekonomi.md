# 01 — Makroekonomi

## Konunun Özeti
Makroekonomi; faiz, enflasyon, büyüme, para ve maliye politikası gibi ekonominin
bütününü ilgilendiren dinamikleri inceler. Yatırımcı için önemi: varlık fiyatlarının
"iskonto oranı" (faiz) ve "nakit akışı beklentileri" (büyüme/enflasyon) makro
değişkenlerle belirlenir. Ray Dalio'nun ifadesiyle piyasalar dört temel güce göre
fiyatlanır: büyüme ↑/↓ ve enflasyon ↑/↓ beklentilerindeki değişimler.

## Temel Kavramlar

### Faiz
- **Politika faizi:** Merkez bankasının bankalara uyguladığı referans oran (Fed funds,
  TCMB 1 hafta repo). Tüm varlık fiyatlamasının "risksiz oran" çıpası.
- **Nominal vs. reel faiz:** Reel faiz ≈ nominal faiz − beklenen enflasyon (Fisher denklemi).
  Varlık fiyatları için asıl önemli olan **reel** faizdir.
- **Verim eğrisi (yield curve):** Farklı vadelerdeki tahvil getirilerinin grafiği.
  Ters dönmüş eğri (kısa vade > uzun vade) tarihsel olarak resesyon öncüsüdür
  (Estrella & Mishkin, 1996). Ancak 2022–2024 ABD tersine dönüşü uzun süre
  resesyonsuz kaldı — gösterge şaşmaz değildir. `[TARTIŞMALI]`

### Enflasyon / Deflasyon
- **Enflasyon:** Genel fiyat düzeyinin sürekli artışı. Ölçümü: TÜFE/CPI (tüketici),
  ÜFE/PPI (üretici), PCE (Fed'in tercih ettiği gösterge), çekirdek (gıda-enerji hariç).
- **Talep enflasyonu** (aşırı ısınan ekonomi) vs. **maliyet enflasyonu** (arz şoku,
  enerji). Politika tepkisi farklıdır: talep enflasyonuna faiz etkili, arz şokuna sınırlı etkili.
- **Deflasyon:** Fiyatların genel düşüşü. Borç yükünü reel olarak artırır (Fisher'ın
  "debt-deflation" teorisi, 1933). Japonya 1990'lar–2010'lar başlıca örnek.
- **Dezenflasyon:** Enflasyonun hızının düşmesi (fiyatlar hâlâ artar, daha yavaş).
- **Stagflasyon:** Düşük büyüme + yüksek enflasyon (ABD 1970'ler).

### Resesyon / Depresyon
- **Resesyon:** Yaygın pratik tanım: 2 çeyrek üst üste negatif GSYH. Resmî ABD tanımı
  NBER'e aittir ve tek kritere bağlı değildir (istihdam, gelir, üretim, satışların
  yaygın ve süreli düşüşü).
- **Depresyon:** Resmî tanımı yoktur; çok derin ve uzun resesyon (GSYH'de ~%10+
  daralma veya yıllarca süren küçülme). Örnek: 1929–1933 Büyük Buhran (ABD GSYH ~%30↓,
  işsizlik ~%25).

### Para Politikası
- **Araçlar:** Politika faizi, zorunlu karşılıklar, açık piyasa işlemleri (APİ),
  niceliksel genişleme/daralma (QE/QT), ileriye dönük yönlendirme (forward guidance),
  makroihtiyati tedbirler.
- **QE:** Merkez bankasının tahvil alarak bilançosunu büyütmesi → uzun vadeli faizleri
  bastırır, risk iştahını artırır. 2008 sonrası Fed bilançosu ~0,9 → ~4,5 trilyon $;
  pandemi döneminde ~9 trilyon $ zirve. `[EĞİTİM VERİSİ — ~Ocak 2026 kesiti]`
- **Aktarım mekanizması:** Faiz → kredi koşulları → tüketim/yatırım → istihdam →
  enflasyon. Gecikme tipik olarak 12–24 ay ("long and variable lags", Friedman).

### Maliye Politikası
- Devletin harcama ve vergi kararları. Genişletici (açık verme) vs. sıkılaştırıcı.
- **Çarpan etkisi:** 1 birim kamu harcamasının GSYH'ye etkisi; resesyonda ve faizin
  sıfıra yakın olduğu ortamlarda daha yüksek (IMF, Blanchard & Leigh 2013: kriz sonrası
  çarpanlar öngörülenden büyüktü).
- Para–maliye etkileşimi: aynı yönde çalışırsa güçlü (2020–21), zıt yönde çalışırsa
  (genişletici maliye + sıkı para, ABD 2023–25) faizler üzerinde yukarı baskı yaratır.

### Merkez Bankaları
- Başlıca aktörler: **Fed** (ikili misyon: fiyat istikrarı + tam istihdam, hedef %2 PCE),
  **ECB** (%2 simetrik hedef), **BoJ** (uzun yıllar YCC, 2024'te negatif faizden çıkış),
  **PBoC**, **TCMB** (%5 orta vadeli hedef, fiilen çok üzerinde).
- **Bağımsızlık:** Ampirik literatür merkez bankası bağımsızlığı ile düşük enflasyon
  arasında güçlü ilişki bulur (Alesina & Summers 1993). Türkiye 2019–2023 dönemi,
  bağımsızlık erozyonunun enflasyonist sonucuna güncel örnek olarak gösterilir. `[TARTIŞMALI — nedensellik tartışılır, korelasyon güçlü]`

### Tahvil Piyasaları
- **Fiyat–getiri ters ilişkisi:** Faiz yükselirse mevcut tahvilin fiyatı düşer.
- **Süre (duration):** Faiz duyarlılığı ölçüsü. Duration 7 olan tahvil, faizde +1 puan
  için yaklaşık %7 değer kaybeder (konveksite düzeltmesi hariç).
- **Kredi spreadi:** Şirket tahvili getirisi − hazine getirisi; risk iştahı göstergesi.
- ABD Hazine piyasası (~28+ trilyon $) küresel "risksiz oran"ı belirler; 10 yıllık
  Treasury getirisi tüm dünyada varlık fiyatlamasının referansıdır.

### Döviz Piyasaları
- Günlük hacmi ~7,5 trilyon $ ile dünyanın en büyük piyasası (BIS Triennial Survey 2022).
- Belirleyiciler: faiz farkları (carry), enflasyon farkları (satın alma gücü paritesi —
  uzun vade), cari denge, sermaye akışları, risk iştahı (risk-off'ta USD/JPY/CHF güçlenir).
- **Carry trade:** Düşük faizli para ile borçlanıp yüksek faizliye yatırım. Sakin
  dönemlerde kârlı, şoklarda ani tersine dönüş ("merdivenle çıkıp asansörle inmek").
  Örnek: Ağustos 2024 yen carry çözülmesi, Nikkei'de tek günde %12'lik düşüşü tetikledi.

### Likidite ve Küresel Sermaye Akışları
- **Likidite:** (1) piyasa likiditesi — varlığı fiyatı bozmadan alıp satabilme;
  (2) fonlama likiditesi — borç bulabilme. Krizlerde ikisi birlikte kurur
  (Brunnermeier & Pedersen 2009).
- Küresel dolar likiditesi sıkılaşınca (güçlü USD + yüksek Fed faizi) gelişmekte olan
  ülkelerden sermaye çıkışı yaşanır ("taper tantrum" 2013, "Fragile Five").

## Nasıl Çalışır? (Yatırımcı Perspektifinden Aktarım Zinciri)
1. Merkez bankası faizi değiştirir → risksiz oran değişir.
2. İskonto oranı değişir → tüm varlıkların bugünkü değeri yeniden fiyatlanır
   (uzun "duration"lı varlıklar — büyüme hisseleri, uzun tahviller — en duyarlı).
3. Kredi koşulları değişir → şirket kârları ve temerrüt oranları etkilenir.
4. Döviz kurları faiz farklarına göre ayarlanır → ihracatçı/ithalatçı kârlılığı değişir.
5. Beklentiler (forward guidance) çoğu zaman kararın kendisinden daha çok fiyat oynatır:
   piyasalar gerçekleşeni değil, beklenenden sapmayı fiyatlar.

## Avantajlar (Makro Analizin Yatırımcıya Katkısı)
- Varlık sınıfı seçimi ve ağırlıklandırma (asset allocation) için çerçeve sağlar.
- Büyük rejim değişimlerini (enflasyon rejimi, faiz döngüsü) erken fark etmeye yardım eder.
- Kuyruk risklerine (resesyon, kriz) karşı korunma planlaması sağlar.

## Dezavantajlar / Sınırlar
- **Zamanlama neredeyse imkânsızdır:** Makro tahmin isabeti düşüktür; IMF/konsensüs
  tahminleri dönüm noktalarını sistematik olarak ıskalar (Loungani 2001: "resesyonların
  neredeyse hiçbiri önceden tahmin edilemedi").
- Doğru makro görüş bile yanlış piyasa pozisyonuna dönüşebilir (piyasa zaten fiyatlamıştır).
- Aşırı makro odak, hisse seçiminde fırsat maliyeti yaratır. Peter Lynch: "Faizleri
  tahmin etmeye 13 dakikadan fazla harcadıysanız 10 dakikanızı boşa harcamışsınızdır."

## Riskler
- **Enflasyon rejim değişimi:** 2021–22'de "geçici" (transitory) diyen konsensüs yanıldı;
  tahvil yatırımcıları 2022'de tarihî kayıp yaşadı (Bloomberg US Agg −%13, 10y+ Treasury −%29 civarı). `[EĞİTİM VERİSİ]`
- **Politika hatası:** Çok erken gevşeme (1970'ler Burns Fed'i) veya çok geç sıkılaşma.
- **Borç sarmalı / mali baskınlık (fiscal dominance):** Kamu borcu faiz artışını
  taşıyamaz hale gelirse merkez bankası enflasyonla mücadelede kısıtlanır. `[TARTIŞMALI — ABD için güncel tartışma konusu]`
- **Jeopolitik şoklar:** Enerji fiyatı şokları hem enflasyonu artırır hem büyümeyi düşürür;
  para politikasını ikileme sokar (2022 Ukrayna, 2026 Orta Doğu).

## Gerçek Örnekler
- **Volcker şoku (1979–82):** Fed faizi ~%20'ye çıkardı; iki resesyon pahasına %14
  enflasyon %3'e indi. Ders: kararlı sıkılaşma çalışır ama bedeli ağırdır.
- **Japonya (1990–):** Varlık balonu patlaması + deflasyon + sıfır faiz + QE öncülüğü.
  "Balance sheet recession" kavramı (Richard Koo): özel sektör borç öderken para
  politikası etkisizleşir.
- **Türkiye (2021–23):** Enflasyon %85'e (Ekim 2022, TÜİK) yükselirken faiz indirimi
  denemesi; ardından Haziran 2023 sonrası ortodoksiye dönüş, politika faizi %8,5'ten
  %50'ye (Mart 2024). 2025–26'da kademeli indirim döngüsü, Haziran 2026'da %37'de duraklama
  `[DOĞRULANDI — TCMB duyuruları, 2026-06]`.
- **2022 küresel sıkılaşma:** Fed 0 → %5,25–5,50 (en hızlı döngülerden biri); hem hisse
  hem tahvil aynı yıl düştü — "60/40 portföyün en kötü yıllarından biri".

## Tarihsel Olaylar (Kronolojik Mini Zaman Çizelgesi)
- 1929–33 Büyük Buhran → para arzının daralmasının rolü (Friedman & Schwartz 1963).
- 1971 Bretton Woods'un sonu → fiat para dönemi.
- 1973/79 petrol şokları → stagflasyon.
- 1987 Kara Pazartesi → portföy sigortası kaynaklı mekanik satış.
- 1997–98 Asya/Rusya krizleri → sabit kur + kısa vadeli dış borç kırılganlığı; LTCM çöküşü.
- 2008 Küresel Finansal Kriz → kaldıraç + konut balonu + gölge bankacılık; QE çağı başladı.
- 2010–12 Euro borç krizi → "whatever it takes" (Draghi, Temmuz 2012).
- 2020 COVID → tarihin en hızlı ayı piyasası ve en büyük ortak para+maliye genişlemesi.
- 2021–23 küresel enflasyon dalgası ve agresif sıkılaşma.
- 2024–26 kademeli normalleşme; 2026'da enerji şokuyla enflasyonun yeniden yükselişi
  `[DOĞRULANDI — bkz. 13-guncel-durum-2026.md]`.

## En Yaygın Hatalar
1. Tek göstergeye (ör. verim eğrisi) mekanik güven.
2. "Bu sefer farklı" (Reinhart & Rogoff'un kitabının başlığı bilinçli ironidir) — ya da
   tersi hata: her döngünün aynı olacağını sanmak.
3. Nominal ve reel değişkenleri karıştırmak (yüksek nominal faiz ≠ sıkı politika;
   reel faize bakılır).
4. Makro görüşü doğrudan pozisyona çevirmek: "resesyon geliyor → hisse sat" zinciri,
   fiyatlamada ne olduğunu hesaba katmaz.
5. Merkez bankası söylemini değil yalnızca kararını izlemek.
6. Gecikmeleri unutmak: son faiz artışının etkisi 1–2 yıl sonra görülür.

## Uzman Görüşleri (Karşılaştırmalı)
- **Ray Dalio:** Borç döngüleri (kısa ~5–8 yıl, uzun ~75–100 yıl) makroyu belirler;
  "Big Debt Crises" çerçevesi. Çeşitlendirilmiş "All Weather" yaklaşımını savunur.
- **Warren Buffett / Peter Lynch:** Makro tahmine dayalı yatırım kararına karşı;
  şirket odaklı kalın. Buffett: "Tahminler tahmincinin geleceği hakkında çok şey söyler,
  geleceğin kendisi hakkında az şey."
- **George Soros:** Refleksivite — piyasa fiyatları temelleri sadece yansıtmaz, onları
  değiştirir; makro dengesizlikler tek yönlü büyük pozisyon fırsatı yaratır
  (1992 GBP, ~1 milyar $ kâr).
- **Howard Marks:** Tahmin etmeye değil "nerede olduğumuzu bilmeye" (döngü konumu)
  odaklan; "Mastering the Market Cycle".
- **Çelişki notu:** Dalio/Soros makroyu merkeze koyar; Buffett/Lynch neredeyse yok sayar.
  İkisi de uzun dönemde başarılı olmuştur → makro analiz gerekli değil ama bazı
  stratejiler için yeterli olabilir. Strateji ile analiz çerçevesi tutarlı olmalıdır.

## Akademik Çalışmalar
- Friedman & Schwartz (1963), *A Monetary History of the United States* — para arzı ve Buhran.
- Fisher (1933), "The Debt-Deflation Theory of Great Depressions", *Econometrica*.
- Estrella & Mishkin (1996), verim eğrisi–resesyon ilişkisi, NY Fed.
- Alesina & Summers (1993), merkez bankası bağımsızlığı–enflasyon, *JMCB*.
- Reinhart & Rogoff (2009), *This Time Is Different* — 800 yıllık kriz verisi
  (2010 "%90 borç eşiği" makalesindeki hesap hatası da bilinmelidir — Herndon vd. 2013).
- Brunnermeier & Pedersen (2009), "Market Liquidity and Funding Liquidity", *RFS*.
- Bernanke (1983), kredi kanalının Buhran'daki rolü, *AER*.

## Kaynaklar
- Fed: federalreserve.gov (FOMC açıklamaları, SEP/dot plot, H.4.1 bilanço)
- TCMB: tcmb.gov.tr (PPK kararları, enflasyon raporu), TÜİK (TÜFE)
- BIS (bis.org), IMF WEO, FRED (fred.stlouisfed.org)
- Kitaplar: Dalio *Principles for Navigating Big Debt Crises*; Koo *The Holy Grail of
  Macroeconomics*; Marks *Mastering the Market Cycle*

## Güncel Gelişmeler (Temmuz 2026 itibarıyla — kaynaklar için bkz. dosya 13)
- Fed: %3,50–3,75 aralığında, dört toplantıdır sabit; Orta Doğu kaynaklı enerji şoku
  nedeniyle şahin duruş, 2026 sonu medyan beklenti %3,8 (indirim değil olası artırım). `[DOĞRULANDI]`
- ABD TÜFE: Mayıs 2026'da yıllık %4,2 (enerji +%23,5 y/y), çekirdek %2,9. `[DOĞRULANDI]`
- TCMB: politika faizi %37; enerji şoku indirim döngüsünü durdurdu. `[DOĞRULANDI]`

## Sonuç
Makroekonomi, varlık fiyatlarının "hava durumu"dur: yön tayini için gereklidir ama
nokta tahmin aracı değildir. En sağlam kullanım biçimi: (1) rejimi teşhis et
(enflasyon/büyüme/likidite), (2) portföyü rejimlere dayanıklı kur, (3) uç fiyatlamalarda
(döngü aşırılıklarında) kademeli ayarlama yap. Tahmine değil hazırlığa yatırım yap.

## Güven Seviyesi
- Kavramsal çerçeve ve tarihsel olaylar: **%90**
- Ampirik ilişkiler (verim eğrisi, bağımsızlık-enflasyon vb.): **%75** (bağlama bağlı)
- Temmuz 2026 güncel verileri: **%85** (web ile doğrulandı; ikincil kaynaklar içeriyor)
- Eğitim verisine dayalı 2024–25 detayları: **%70** (güncel teyit yok)
