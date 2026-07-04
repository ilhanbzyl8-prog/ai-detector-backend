# 08 — Türev Ürünler

## Konunun Özeti
Türevler, değeri bir dayanak varlıktan türeyen sözleşmelerdir (opsiyon, vadeli işlem,
swap, forward). İki meşru işlevi vardır: **riski transfer etmek (hedge)** ve **görüşü
sermaye-verimli ifade etmek**. Aynı araçlar kaldıraç yoluyla hızlı servet yıkımının da
ana aracıdır. Buffett'ın "finansal kitle imha silahları" uyarısı (2002) ile türev
piyasalarının günlük ekonomik işlevi (küresel nominal büyüklük ~600+ trilyon $, BIS)
arasındaki gerilim, konunun özetidir.

## Temel Kavramlar

### Forward ve Futures (Vadeli İşlemler)
- **Forward:** İki taraf arasında özel (OTC) sözleşme; ileri tarihte, bugünden
  belirlenen fiyatla alım/satım. Karşı taraf riski taşır.
- **Futures:** Borsada standartlaştırılmış forward; takas odası (clearing house)
  karşı taraf riskini üstlenir, günlük marj (mark-to-market) uygulanır.
- Fiyatlama: F = S·e^{(r+depolama−temettü/verim)T} (taşıma maliyeti modeli).
- **Contango** (vadeli > spot) / **Backwardation** (vadeli < spot): roll getirisinin
  yönünü belirler — emtia ETF'lerinin gizli maliyet/getiri kaynağı.
- Kullanım: endeks maruziyeti, emtia hedge'i (havayolunun yakıt hedge'i), kur hedge'i
  (ihracatçının forward satışı — Türkiye pratiğinde yaygın).

### Swap
- Nakit akışı takası. Türler: **faiz swapı** (sabit ↔ değişken; en büyük OTC piyasa),
  **çapraz kur swapı**, **CDS** (kredi temerrüt swapı — temerrüt sigortası; 2008'de
  AIG'nin çöküş nedeni: sattığı CDS'lerin teminat çağrıları), **toplam getiri swapı**
  (TRS — Archegos'un gizli kaldıraç aracı).
- 2021 sonrası LIBOR'dan SOFR/€STR gibi risksiz referans oranlara geçiş tamamlandı.

### Opsiyonlar
- **Call:** Belirli fiyattan alma hakkı (yükümlülük değil). **Put:** Satma hakkı.
- Avrupa (sadece vadede) / Amerikan (her an) kullanım tipi.
- **Fiyat = içsel değer + zaman değeri.** Black-Scholes-Merton (1973) modeli;
  girdiler içinde tek bilinmeyen volatilite → piyasa opsiyonu fiyatlar, model
  **zımni volatiliteyi (IV)** çıkarır. Opsiyon ticareti büyük ölçüde volatilite ticaretidir.
- **Yunanlar:** Delta (fiyat duyarlılığı), Gamma (deltanın değişimi), Theta (zaman
  erimesi), Vega (IV duyarlılığı), Rho (faiz).
- **Volatilite gülümsemesi:** 1987 sonrası endeks putlarında kalıcı IV yükseği —
  piyasa kuyruk riskini Black-Scholes'tan pahalı fiyatlar.
- **Varyans risk primi:** Zımni volatilite, gerçekleşenden ortalamada yüksektir →
  sistematik opsiyon satıcısı prim toplar (sigortacı ekonomisi), ama negatif çarpıklık
  taşır: nadir günlerde yıllar birikimini geri verir (Şub 2018, Mar 2020).

### Temel Opsiyon Stratejileri
- **Covered Call:** Hisse + call satışı. Getiri: prim + sınırlı yükseliş. Yatay/hafif
  yükselen piyasada iyi; güçlü boğada fırsat maliyeti büyük. CBOE BXM endeksi uzun
  dönemde S&P'ye yakın getiriyi daha düşük oynaklıkla verdi. `[EĞİTİM VERİSİ]`
- **Protective Put:** Hisse + put alımı = sigorta. Maliyet sürüklemesi yıllık %2–5
  olabilir; sürekli taşınması pahalıdır (dosya 06'daki tail-hedge tartışması).
- **Collar:** Put al + call sat → düşük maliyetli bant koruması.
- **Straddle:** Aynı kullanım fiyatlı call+put alımı → büyük hareket bekle (yön yok).
  Kâr eşiği: prim toplamı kadar hareket. Kazanç olayları öncesi IV şişkinliğine dikkat.
- **Strangle:** Farklı kullanım fiyatlı call+put → daha ucuz, daha geniş eşik.
- **Iron Condor:** OTM put spread + OTM call spread satışı → dar bantta kalma bahsi;
  sınırlı kâr / sınırlı (ama kârdan büyük) zarar. Perakende "gelir stratejisi" olarak
  pazarlanır; gerçekte negatif çarpık varyans primi hasadıdır — boyutlandırma kritiktir.

## Nasıl Çalışır? (Ekonomik Mantık)
- Türev piyasa **sıfır toplamlıdır** (komisyon sonrası negatif): birinin kârı diğerinin
  zararıdır. Yine de rasyoneldir çünkü taraflar farklı şeyler için oradadır: hedger risk
  satar (bedel ödemeye razı), spekülatör risk alır (prim talep eder), arbitrajcı
  fiyat tutarlılığı sağlar.
- Kaldıraç içseldir: küçük teminat büyük nominal kontrol eder → hem sermaye
  verimliliği hem yıkım potansiyeli.

## Avantajlar
- Hassas risk şekillendirme: sadece istenen riski al/sat (yön, volatilite, vade, seviye).
- Sermaye verimliliği ve kısa pozisyon kolaylığı.
- Tanımlı-risk yapılar (spread'ler) maksimum kaybı baştan sabitler.
- Portföy sigortası mümkün kılar (endeks putu, collar).

## Dezavantajlar
- Karmaşıklık: yunanlar, IV, vade yapısı — hata alanı geniş.
- Zaman erimesi (theta) alıcı aleyhine sürekli çalışır; çoğu perakende opsiyon alımı
  vadede değersiz biter. `[DOĞRULANMADI — "%90'ı değersiz biter" mitinin kesin oranı
  tartışmalıdır; büyük kısmının vadeden önce kapatıldığı bilinir]`
- İşlem maliyeti ve spread'ler likit olmayan serilerde yüksektir.
- Vergi ve muhasebe karmaşası (ülkeye göre değişir).

## Riskler
- **Kaldıraç + gamma:** Kısa opsiyon pozisyonları küçük hareketlerde kârlı, büyük
  harekette üstel zarar (kısa gamma). "Prim toplama" stratejilerinin ölüm nedeni.
- **Likidite/marj sarmalı:** Zarar → marj çağrısı → zorunlu kapama → fiyatı daha da
  aleyhine itme (LTCM, Archegos).
- **Karşı taraf riski** (OTC'de) ve **baz riski** (hedge aracı ile korunacak varlığın
  ayrışması — Metallgesellschaft 1993: uzun vadeli taahhüdü kısa vadeli futures'la
  hedge etmenin nakit akışı riski).
- **Pin/atama riski:** Amerikan opsiyonlarda erken atama; vadede kullanım belirsizliği.

## Gerçek Örnekler
- **1987:** Endeks türevleri + portföy sigortası çöküşü büyüttü; kalıcı miras: put skew.
- **Barings (1995):** Leeson'ın Nikkei futures + kısa straddle pozisyonları 233 yıllık
  bankayı batırdı. Ders: operasyonel kontrol, tek kişilik risk.
- **AIG (2008):** CDS satışı = "asla ödenmeyecek sigorta primi toplama" sanısı;
  180 mlr $ kamu kurtarması.
- **Şubat 2018 "Volmageddon":** Kısa-VIX ürünleri bir günde yok oldu.
- **GameStop (Ocak 2021):** Perakende call alımı → market maker delta hedge'i →
  "gamma squeeze"; opsiyon akışının spot fiyatı sürükleyebildiğinin kanıtı.
- **0DTE dalgası (2023–):** S&P opsiyon hacminin yarısına yakını aynı-gün vadeli;
  piyasa mikro yapısını değiştirdi, sistemik etkisi hâlâ tartışılıyor. `[EĞİTİM VERİSİ]`

## Tarihsel Olaylar
- MÖ ~600: Thales'in zeytin presi opsiyonu (Aristoteles) — kavramın antikliği.
- 1630'lar: Lale forward'ları; 1848: CBOT; 1973: CBOE + Black-Scholes yayını
  (1997 Nobel: Scholes & Merton); 1998: LTCM (aynı Nobelliler); 2000'ler: CDS patlaması;
  2010 Dodd-Frank: OTC türevlere merkezi takas zorunluluğu.

## En Yaygın Hatalar
1. Kaldıracı pozisyon boyutuyla karıştırmak (10x kaldıraç = 1/10 boyut kuralını unutmak).
2. IV yüksekken opsiyon almak (olay sonrası "volatility crush" ile yön doğru, para kayıp).
3. Kısa opsiyonda "küçük ve istikrarlı gelir"e alışıp kuyruk gününe boyut büyütmüş girmek.
4. Likit olmayan serilerde piyasa emri kullanmak.
5. Atama/erken kullanım mekaniğini bilmemek (temettü öncesi call ataması).
6. Hedge'i kâr merkezi sanmak: hedge maliyettir, sigortadır; kâr etmesi beklenmez.
7. Baz riskini görmemek (korunan varlıkla türevin dayanağı farklıysa).

## Uzman Görüşleri
- **Buffett (2002 mektubu):** Sistemik zincir riski uyarısı; ama Berkshire uzun vadeli
  endeks putu **satmıştır** — karşı çıktığı araç değil, şeffaf olmayan kaldıraç zinciridir.
- **Taleb (eski opsiyon trader'ı):** Dışbükeylik al, içbükeylik satma; nadir olay
  sigortası sistematik ucuz değil pahalı **görünür** ama yanlış fiyatlanmış kuyruklar bulunabilir.
- **Sıradan karşı görüş (varyans primi literatürü):** Sistematik opsiyon satışı uzun
  vadede pozitif Sharpe üretmiştir (BXM, PutWrite endeksleri) — Taleb'in tam tersi
  pozisyonun da akademik savunusu vardır. **İki taraf da veriyle konuşur; fark,
  kuyruk gününde hayatta kalacak boyutlandırmadadır.** `[TARTIŞMALI — çekirdek tartışma]`
- **Hull (ders kitabı otoritesi):** Araç nötrdür; risk yönetimi çerçevesi belirleyicidir.

## Akademik Çalışmalar
- Black & Scholes (1973), *JPE*; Merton (1973), *Bell Journal*.
- Carr & Wu; Bakshi & Kapadia — varyans risk primi literatürü.
- Bollen & Whaley (2004), opsiyon talebinin IV eğrisine etkisi, *JF*.
- Figlewski — opsiyon piyasa etkinliği çalışmaları.
- BIS OTC türev istatistikleri (yarıyıllık, bis.org) — piyasa büyüklüğü birincil kaynağı.

## Kaynaklar
- Hull, *Options, Futures, and Other Derivatives* (standart ders kitabı)
- Natenberg, *Option Volatility and Pricing* (uygulayıcı standardı)
- Taleb, *Dynamic Hedging* (ileri düzey)
- CBOE/CME eğitim portalları; VIOP (Borsa İstanbul vadeli işlem ve opsiyon piyasası) dokümanları

## Güncel Gelişmeler
- 0DTE hacminin kalıcılaşması ve perakende opsiyon akışının fiyat oluşumundaki rolü
  düzenleyici gündemde. `[EĞİTİM VERİSİ]`
- Kripto türevleri (perpetual futures) spot piyasadan büyük; funding rate mekanizması
  klasik vadeli fiyatlamanın kripto uyarlaması. `[EĞİTİM VERİSİ]`

## Sonuç
Türevler risk **yaratmaz**, riski **yeniden dağıtır** — kime ne kadar gittiğini bilen
için araç, bilmeyen için tuzaktır. Bireysel kullanım için güvenli hiyerarşi:
(1) tanımlı-risk yapılar (spread, collar), (2) nakit teminatlı satışlar,
(3) yalın kaldıraçlı kısa pozisyonlardan uzak durmak. Her opsiyon pozisyonu bir
volatilite görüşüdür; yön görüşü sanmak en pahalı yanılgıdır.

## Güven Seviyesi
- Mekanik ve fiyatlama: **%95**
- Vaka çalışmaları: **%90**
- Varyans primi tartışması: **%75** (iki taraflı literatür)
- Perakende istatistik mitleri: **%50 — işaretlendi**
