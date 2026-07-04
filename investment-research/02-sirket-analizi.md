# 02 — Şirket (Temel) Analizi

## Konunun Özeti
Temel analiz, bir şirketin finansal tablolarından ve iş modelinden yola çıkarak
gerçek (içsel) değerini tahmin etme disiplinidir. Benjamin Graham'ın kurduğu,
Buffett ve Munger'ın "harika şirketi makul fiyata al" biçiminde evrimleştirdiği
yaklaşımın çekirdeğidir. Üç finansal tablo birlikte okunur; tek başına hiçbiri yeterli değildir.

## Temel Kavramlar

### Üç Finansal Tablo
1. **Gelir Tablosu (Income Statement):** Dönemsel performans. Hasılat → brüt kâr →
   faaliyet kârı (EBIT) → net kâr. Tahakkuk esaslıdır: kâr ≠ nakit.
2. **Bilanço (Balance Sheet):** Anlık fotoğraf. Varlıklar = Yükümlülükler + Özkaynak.
   Kalitesi önemlidir: şerefiye (goodwill) ve maddi olmayan varlıklar şişkin olabilir.
3. **Nakit Akış Tablosu (Cash Flow Statement):** İşletme, yatırım ve finansman
   faaliyetlerinden nakit. Manipülasyona en dirençli tablo; "kâr görüştür, nakit gerçektir".

### Marjlar
- **Brüt marj** = Brüt kâr / Hasılat → ürün/fiyatlama gücü.
- **Faaliyet (EBIT) marjı** → operasyonel verimlilik.
- **Net marj** → tüm maliyetler sonrası kârlılık.
- **FCF marjı** = Serbest nakit akışı / Hasılat → gerçek nakit üretme gücü.
- Sektörler arası karşılaştırma yanıltıcıdır: yazılımda %80 brüt marj normal,
  perakendede %25 iyi olabilir. Marjın **yönü ve istikrarı** seviyesi kadar önemlidir.

### Kârlılık Oranları
- **ROE** = Net kâr / Ortalama özkaynak. DuPont ayrıştırması:
  ROE = Net marj × Varlık devri × Kaldıraç. Yüksek ROE'nin kaynağı önemlidir —
  kaldıraçla şişirilmiş ROE düşük kaliteli sinyaldir.
- **ROA** = Net kâr / Toplam varlık → varlık verimliliği (bankalarda kritik).
- **ROIC** = NOPAT / Yatırılan sermaye. En önemli kalite göstergesi sayılır:
  **ROIC > WACC** ise şirket değer yaratır, altındaysa büyüme değer yok eder.
  Munger/Buffett'ın "moat" (hendek) kavramının sayısal izdüşümü: yüksek ve **kalıcı** ROIC.

### EBITDA / FAVÖK
- EBITDA = Faiz, vergi, amortisman ve itfa öncesi kâr (Türkçe karşılığı FAVÖK).
- Kullanımı: sermaye yapısından bağımsız karşılaştırma, borç kapasitesi (Net Borç/FAVÖK).
- Eleştirisi — Buffett: "Amortismanı diş perisi mi ödüyor?"; Munger: EBITDA'yı
  "bullshit earnings" diye anar. Yoğun sabit sermaye gerektiren işlerde EBITDA
  gerçek kârlılığı ciddi biçimde abartır. `[TARTIŞMALI — pratikte yaygın, teorik olarak kusurlu]`

### Serbest Nakit Akışı (FCF)
- FCF = İşletme faaliyetlerinden nakit − yatırım harcamaları (capex).
- İnce ayrım: **bakım capex** vs. **büyüme capex** — sadece bakım capex düşülürse
  "sahip kazancı" (owner earnings, Buffett 1986 mektubu) elde edilir; ayrıştırma
  tahmin gerektirir.
- Hisse bazlı ücretlendirme (SBC) nakit çıkışı olmasa da gerçek maliyettir; FCF'e
  geri eklenmemeli veya seyreltme ayrıca hesaba katılmalıdır. `[TARTIŞMALI — analistler farklı uygular]`

### Borç Analizi
- **Net borç** = Toplam finansal borç − nakit ve benzerleri.
- Net Borç/FAVÖK: <1 muhafazakâr, 2–3 tipik, >4 riskli (sektöre göre değişir).
- **Faiz karşılama** = EBIT / Faiz gideri (>3–4 rahat sayılır).
- Vade yapısı ve para birimi uyumu kritik: kısa vadeli döviz borcu + TL geliri =
  kırılganlık (Türk şirketlerinde 2018 kur şokunun ana hasar mekanizması).
- Faaliyet kiralamaları (IFRS 16 sonrası bilançoda) ve emeklilik açıkları "gizli borç".

### İçsel Değer
- Bir işletmenin değeri, ömrü boyunca üreteceği nakit akışlarının bugüne indirgenmiş
  toplamıdır (John Burr Williams, 1938). Hesaplama yöntemleri dosya 03'te.

## Nasıl Çalışır? (Analiz Süreci)
1. **İşi anla:** Şirket parayı nasıl kazanıyor? Müşterisi kim? Fiyatlama gücü var mı?
   (10-K/faaliyet raporunun "iş tanımı" bölümü.)
2. **Kaliteyi ölç:** 5–10 yıllık ROIC, marj istikrarı, hasılat büyümesi, FCF dönüşümü
   (FCF/net kâr oranı sürekli <%70 ise kâr kalitesi sorgulanır).
3. **Bilanço sağlamlığı:** Net borç, vade, karşılıklar, çalışma sermayesi döngüsü.
4. **Kırmızı bayraklar:** Alacakların hasılattan hızlı büyümesi, sık "tek seferlik"
   giderler, agresif gelir tanıma, denetçi değişikliği, yönetici satışları,
   Beneish M-Score / Altman Z-Score taraması.
5. **Yönetim ve sermaye tahsisi:** Geri alım/temettü/satın alma kararlarının geçmişi;
   hissedar mektuplarının dürüstlüğü.
6. **Değerle** (dosya 03) ve **güvenlik marjı** bırak (Graham: değerin belirgin altında öde).

## Avantajlar
- Fiyattan bağımsız bir "değer çıpası" sağlar; balon ve panikte pusula görevi görür.
- Uzun vadede hisse getirisi işletme performansına yakınsar (Buffett: "kısa vadede
  oylama makinesi, uzun vadede tartı" — Graham'a atfen).
- Dolandırıcılık ve iflas riskini erken yakalamada en etkili araç setidir.

## Dezavantajlar
- Zaman yoğundur; bilgi avantajı kurumsal yatırımcıya kaymıştır.
- Kısa vadeli fiyat hareketleri hakkında hiçbir şey söylemez; "değer tuzağı"nda
  yıllarca beklenebilir.
- Girdi varsayımlarına aşırı duyarlıdır (çöp girerse çöp çıkar).
- Muhasebe standartları (UFRS/GAAP) yönetime takdir alanı bırakır; tablolar
  "gerçeğin" kendisi değil, bir temsili ve bazen makyajlı halidir.

## Riskler
- **Muhasebe manipülasyonu/sahtecilik:** Enron (2001, SPV'lerle borç gizleme),
  WorldCom (2002, gideri aktifleştirme), Wirecard (2020, 1,9 milyar € hayali nakit),
  Luckin Coffee (2020, hayali satışlar).
- **Yapısal bozulma:** Sağlam görünen tablolar geriye bakar; iş modeli çökerse tarih
  yanıltır (Kodak, Nokia, gazete şirketleri).
- **Bilgi asimetrisi:** Yönetim her zaman dış yatırımcıdan fazlasını bilir.

## Gerçek Örnekler
- **Apple (Buffett, 2016–):** Yüksek ROIC + ekosistem hendeği + agresif geri alım;
  Berkshire'ın en büyük pozisyonu oldu ve tarihinin en kârlı yatırımlarından biri haline geldi.
- **Amazon paradoksu:** Yıllarca düşük net kâr, güçlü işletme nakit akışı ve büyüme
  capex'i; sadece P/E'ye bakan analiz şirketi 20 yıl "pahalı" diye kaçırdı → tabloları
  iş modeli bağlamında okumanın önemi.
- **Enron dersi:** Kâr artarken işletme nakit akışının zayıflığı ve anlaşılmaz dipnotlar;
  Jim Chanos'un short tezi büyük ölçüde tablo analiziyle kuruldu.

## Tarihsel Olaylar
- 1934: Graham & Dodd, *Security Analysis* — modern temel analizin doğuşu.
- 2001–02: Enron/WorldCom → Sarbanes-Oxley Yasası (CEO/CFO tablo sertifikasyonu).
- 2020: Wirecard → Alman denetim reformu; "denetlenmiş" ibaresinin sınırları.

## En Yaygın Hatalar
1. Tek yılın verisiyle karar (döngüsel şirketlerde zirve kârına düşük P/E ödemek —
   "değer tuzağının" klasiği).
2. Nakit akışını hiç kontrol etmeden kâra güvenmek.
3. EBITDA'yı nakit akışı sanmak.
4. Sektör bağlamı olmadan oran karşılaştırmak.
5. Seyreltmeyi (SBC, varant, dönüştürülebilir tahvil) yok saymak.
6. Dipnotları okumamak — kritik bilgi (dava, taahhüt, ilişkili taraf) çoğu zaman oradadır.
7. Hikâyeye âşık olup sayıları hikâyeye uydurmak (onay önyargısı, bkz. dosya 09).

## Uzman Görüşleri
- **Graham:** Nicel sağlamlık + güvenlik marjı; "net-net" (NCAV altı) hisseler.
- **Buffett/Munger:** Nitelik > ucuzluk. "Vasat şirketi harika fiyata almaktansa
  harika şirketi makul fiyata al" (Munger'ın etkisi).
- **Peter Lynch:** "Bildiğin şeyi al" + PEG; kategorilere göre analiz
  (stalwart, fast grower, cyclical, turnaround…). *One Up on Wall Street*.
- **Damodaran:** Hikâye + sayı birlikte; sadece oran ezberi değil, anlatının
  sayısal tutarlılığı (*Narrative and Numbers*).
- **Çelişki:** Derin değer (Graham tarzı istatistiksel ucuzluk) vs. kalite/bileşik
  büyütücü (compounder) ekolü. Akademik veri her iki primi de dönemsel bulur;
  2010'lar kalite/büyümenin, 1970–80'ler derin değerin lehineydi.

## Akademik Çalışmalar
- Sloan (1996), "accrual anomaly": tahakkuku yüksek (nakde dönüşmeyen kârlı) şirketler
  sonraki dönemde düşük getiri sağlar, *The Accounting Review*.
- Piotroski (2000), F-Score: 9 maddelik temel sağlamlık skoru ucuz hisselerde getiriyi
  belirgin iyileştirir, *Journal of Accounting Research*.
- Beneish (1999), M-Score: kâr manipülasyonu tespit modeli.
- Altman (1968), Z-Score: iflas tahmini.
- Novy-Marx (2013), brüt kârlılık (gross profitability) primi, *JFE*.

## Kaynaklar
- SEC EDGAR (10-K, 10-Q), KAP (kap.org.tr — Türk şirketleri)
- Graham & Dodd, *Security Analysis*; Graham, *Akıllı Yatırımcı*
- Buffett'ın hissedar mektupları (berkshirehathaway.com — birincil kaynak, ücretsiz)
- Damodaran'ın NYU sayfası (pages.stern.nyu.edu/~adamodar) — veri setleri ve dersler
- Schilit, *Financial Shenanigans* — manipülasyon vakaları

## Güncel Gelişmeler
- Yapay zekâ araçları tablo taramayı hızlandırıyor; ancak dipnot/nitel okumada insan
  yargısı hâlâ belirleyici. `[EĞİTİM VERİSİ]`
- 2026 enerji şoku, enerji-yoğun sektörlerde marj baskısını yeniden gündeme getirdi
  (bkz. dosya 13). `[DOĞRULANDI — makro veri düzeyinde]`

## Sonuç
Şirket analizi = kalite (ROIC, marj istikrarı, bilanço) + kâr kalitesi (nakit dönüşümü)
+ yönetim (sermaye tahsisi) + fiyat (dosya 03). Tablolar arası çapraz kontrol ve
dipnot okuma, orandan daha fazla bilgi taşır. Nakit akışı en güvenilir sinyaldir.

## Güven Seviyesi
- Muhasebe/oran tanımları: **%95**
- Analiz süreci ve kırmızı bayraklar: **%85**
- Vaka detayları (Enron, Wirecard vb.): **%90** (iyi belgelenmiş)
- Ekol karşılaştırmaları: **%75** (yorum içerir)
