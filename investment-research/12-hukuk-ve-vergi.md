# 12 — Hukuk, Regülasyon ve Vergi

> **ÖZEL UYARI:** Bu modül, bilgi tabanının en hızlı eskiyen ve en yüksek hata riskli
> alanıdır. Vergi oranları ve mevzuat sık değişir; ülke ve kişisel duruma göre farklılaşır.
> Buradaki hiçbir şey vergi/hukuk danışmanlığı değildir — **uygulamadan önce güncel
> resmî kaynak ve yetkili danışman şarttır.** Bu dosyadaki Türkiye vergi detayları
> `[DOĞRULANMADI — güncel mevzuat teyidi gerekli]` varsayılmalıdır.

## Konunun Özeti
Yatırımcının hukuki çevresi üç katmandır: (1) piyasanın düzenlenmesi (manipülasyon,
içeriden öğrenme, halka arz kuralları), (2) yatırımcının korunması (aracı kurum
yükümlülükleri, tazmin sistemleri, uygunluk testleri), (3) vergilendirme. Getiri
hesabında vergi ve maliyet, çoğu yatırımcının modellemediği ama sonucu belirleyen değişkendir.

## Temel Kavramlar

### Düzenleyici Otoriteler
- **ABD:** SEC (menkul kıymetler), CFTC (türev/emtia), FINRA (aracı kurum öz-düzenleme),
  Fed/OCC/FDIC (bankacılık). Kripto yetki paylaşımı SEC-CFTC arasında yıllardır
  tartışmalı; CLARITY Act bu sınırı çizmeyi amaçlıyor (Temmuz 2026: Senato aşamasında
  `[DOĞRULANDI — dosya 13]`).
- **AB:** ESMA + ulusal otoriteler; MiFID II (yatırımcı koruması, şeffaflık),
  UCITS (fon standardı), MiCA (kripto — 2024'ten itibaren uygulamada).
- **Türkiye:** SPK (sermaye piyasaları — 6362 sayılı SPKn), BDDK (bankalar),
  TCMB (ödeme sistemleri, kur), MASAK (aklama ile mücadele). Borsa İstanbul + Takasbank
  + MKK piyasa altyapısı. Yatırımcı Tazmin Merkezi (YTM): aracı kurum iflasında
  belirli limitte tazmin (limit yıllara göre güncellenir `[DOĞRULANMADI]`).

### Piyasa Suçları
- **İçeriden öğrenenlerin ticareti (insider trading):** Kamuya açıklanmamış bilgiyle
  işlem. ABD'de hapis dahil ağır yaptırım (Rajaratnam/Galleon 2011: 11 yıl); Türkiye'de
  SPKn m.106 ile suç.
- **Manipülasyon:** Fiyat/hacim yanıltması (pump-and-dump, spoofing, wash trade).
  Türkiye'de sosyal medya üzerinden hisse yönlendirme vakaları SPK'nın süreklilik arz
  eden gündemi; SPK haftalık bültenlerinde işlem yasakları yayımlanır.
- **Piyasa dolandırıcılığı türleri:** Ponzi (Madoff 2008 — 65 mlr $), sahte halka
  arz/token satışı, "finfluencer" yönlendirmeleri.

### Yatırımcı Koruması Mekanizmaları
- Ayrıştırılmış saklama (müşteri varlığı ≠ kurum bilançosu — FTX 2022'nin ihlal ettiği
  temel ilke), tazmin fonları (SIPC-ABD, YTM-TR), uygunluk/yerindelik testleri,
  ürün risk bildirimleri (özellikle kaldıraçlı ürünlerde zorunlu — CFD'lerde
  "hesapların %70+ kaybediyor" uyarıları düzenleyici zorunluluktur).

### Vergilendirme — Kavramsal Çerçeve (ülkeler üstü)
- **Sermaye kazancı vergisi:** Alış-satış farkı üzerinden; çoğu ülkede tutma süresine
  göre farklılaşır (ABD: 1 yıl+ uzun vade indirimli oran).
- **Temettü/faiz stopajı:** Kaynakta kesinti; çifte vergilendirme anlaşmaları (ÇVÖA)
  yurt dışı yatırımda kritik (ör. ABD hisse temettüsünde Türk mukimine %15 W-8BEN ile
  `[DOĞRULANMADI — anlaşma güncel teyidi gerekli]`).
- **Vergi ertelemesi ve verimlilik:** Realize edilmeyen kazanç çoğu sistemde
  vergilenmez → düşük devir hızı, bileşik getiride "vergi ertelemesi" avantajı yaratır
  (Buffett'ın "asla satma" yaklaşımının sessiz vergisel mantığı).
- **Zarar mahsubu (tax-loss harvesting):** Kayıpların kazançlardan düşülmesi;
  ABD'de wash-sale kuralı (30 gün) ile sınırlanır.

### Türkiye Vergi Ana Hatları `[DOĞRULANMADI — tümü için güncel GİB/mevzuat teyidi gerekli; son bilinen çerçeve ~2025]`
- BIST hisse senedi alım-satım kazancı: gerçek kişide stopaj yoluyla vergileme
  (uzun süre %0 uygulandı; oran ve istisnaların güncel durumu mutlaka teyit edilmeli).
- Mevduat faizi, fon, tahvil getirileri: vadeye/ürüne göre değişen stopaj oranları;
  oranlar Cumhurbaşkanı kararlarıyla sık değişmiştir.
- Yurt dışı piyasalardan (ABD hisseleri vb.) elde edilen kazanç: beyan esaslı;
  değer artış kazancı + temettü beyanı gerekebilir.
- Kripto: 2026 itibarıyla Türkiye'de kripto kazançlarına özel vergi rejimi tartışması
  sürüyordu; işlem vergisi/beyan düzenlemesi güncel mevzuattan teyit edilmelidir.
- Enflasyon döneminde nominal kazanç vergilemesi: reel kayıpta bile vergi doğabilir
  (endeksleme kuralları ürüne göre değişir — ÜFE endekslemesi bazı kazançlarda mümkün).

## Nasıl Çalışır? (Yatırımcının Uyum Süreci)
1. Ürünün hukuki statüsünü bil (menkul kıymet mi, türev mi, kripto mu — koruma rejimi değişir).
2. Aracının lisansını doğrula (SPK/SEC lisans sorgusu; lisanssız "forex/kripto"
   platformları en yaygın dolandırıcılık kanalı).
3. Vergisel olayları kaydet: her satış, temettü, temettü reinvest — işlem günlüğü
   beyan dönemini kolaylaştırır.
4. Yıllık beyan takvimini ve eşiklerini takip et; belirsizlikte mali müşavire danış.

## Avantajlar (Regülasyonun Yatırımcıya Faydası)
- Şeffaflık zorunluluğu (finansal tablo, izahname) analiz yapılabilirliğin temelidir.
- Saklama/tazmin mekanizmaları karşı taraf riskini sınırlar.
- Manipülasyon yaptırımları piyasa güvenini (ve likiditeyi) besler.

## Dezavantajlar / Maliyetler
- Uyum maliyeti ve erişim kısıtları (ör. AB'de PRIIPs nedeniyle ABD ETF'lerine
  perakende erişim engeli; Türk yatırımcının bazı ürünlere erişim kısıtları).
- Regülasyon arbitrajı: faaliyet en gevşek bölgeye kayar (offshore kripto borsaları).
- Aşırı düzenlemenin yenilikçiliği yavaşlatması vs. eksik düzenlemenin kriz üretmesi
  arasındaki sarkaç (2008 → Dodd-Frank → 2018 gevşetme → 2023 SVB tartışması). `[TARTIŞMALI]`

## Riskler
- **Mevzuat değişikliği riski:** Vergi oranı/istisna bir kararname ile değişebilir
  (Türkiye'de stopaj oranları defalarca gece yarısı kararlarıyla değişti).
- **Yetkisiz platform riski:** Tazmin dışı kalma; para iadesinin fiilen imkânsızlaşması.
- **Beyan ihmali:** Yurt dışı kazançların beyan edilmemesi cezalı tarhiyatla dönebilir;
  bilgi paylaşım anlaşmaları (CRS) ile görünürlük arttı.
- **Regülasyonla çatışan ürün:** SEC'in menkul kıymet saydığı bir token'ın borsalardan
  kaldırılması fiyatı çökertebilir (2023 dava dalgası örnekleri).

## Gerçek Örnekler / Tarihsel Olaylar
- 1933/34 ABD Menkul Kıymet Kanunları: 1929 sonrası modern düzenlemenin doğuşu.
- 2001 Imar Bankası (TR): kayıt dışı mevduat/hazine bonosu skandalı → Türk denetim
  reformları.
- 2008 Madoff: "SEC denetliyor" varsayımının sınırı — düzenleyici varlığı garanti değildir.
- 2010 Dodd-Frank + Volcker; 2018 MiFID II; 2020 Wirecard (denetçi/denetleyici zafiyeti).
- 2022 FTX: müşteri varlığı ayrıştırma ilkesinin ihlali; kripto regülasyonunu hızlandırdı.
- 2023 SVB: faiz riskinin denetim kapsamı dışında birikmesi.
- 2024–26 ABD kripto çerçevesi: spot ETF onayları (2024), GENIUS Act — stablecoin
  (2025 `[EĞİTİM VERİSİ]`), CLARITY Act süreci (2026 `[DOĞRULANDI — dosya 13]`).

## En Yaygın Hatalar
1. Vergiyi getiri hesabına katmamak (brüt getiriyle plan yapmak).
2. Lisanssız platformda "yüksek kaldıraç/garantili getiri" tekliflerine girmek.
3. Yurt dışı kazanç beyanını unutmak.
4. Ürünün hukuki niteliğini (ve tazmin kapsamını) bilmeden almak.
5. Vergi kuyruğunun yatırım kararını yönetmesine izin vermek (satılması gereken
   pozisyonu vergi korkusuyla tutmak) — vergi optimizasyonu araçtır, amaç değil.
6. "Herkes yapıyor" diye sosyal medya sinyal gruplarına katılmak (manipülasyona
   iştirak hukuki risk doğurabilir).

## Uzman Görüşleri
- **Bogle/Buffett:** Vergi verimliliği uzun vadeli bileşik getirinin sessiz ortağıdır;
  düşük devir = düşük vergi sürtünmesi.
- **Damodaran:** Regülasyon değerlemeye "beklenen değer" olarak katılmalı (senaryo
  ağırlıklandırma), ikili (var/yok) düşünülmemeli.
- **Kripto tartışması:** "Düzenleme meşrulaştırır ve sermaye çeker" (sektör görüşü) vs.
  "düzenleme merkeziyetsizlik vaadini boşaltır" (kripto-yerli görüş). `[TARTIŞMALI]`

## Akademik Çalışmalar
- La Porta, Lopez-de-Silanes, Shleifer & Vishny (1998), "Law and Finance", *JPE* —
  hukuk sisteminin (yatırımcı koruması) piyasa gelişmişliğini belirlemesi.
- Bhattacharya & Daouk (2002), içeriden öğrenme yasalarının uygulanmasının sermaye
  maliyetini düşürmesi, *JF*.
- Poterba (2002), vergilerin yatırım getirisine etkisi literatürü.

## Kaynaklar
- Birincil: mevzuat.gov.tr, GİB (gib.gov.tr), SPK (spk.gov.tr) duyuru/bültenleri,
  SEC (sec.gov, investor.gov), ESMA
- Türkiye pratiği: aracı kurumların yıllık "menkul kıymet gelirlerinin
  vergilendirilmesi" rehberleri (her yıl güncellenir — en pratik ikincil kaynak)

## Güncel Gelişmeler
- ABD: CLARITY Act Senato takviminde; stablecoin çerçevesi yürürlükte. `[DOĞRULANDI/EĞİTİM VERİSİ karışık — dosya 13]`
- Türkiye: kripto ve yüksek enflasyon döneminde stopaj oranlarında sık değişiklik
  riski sürüyor; her beyan dönemi öncesi güncel rehber kontrolü zorunlu. `[Yorum]`

## Sonuç
Hukuk/vergi katmanı "sıkıcı" ama asimetrik risklidir: doğru yapmak küçük avantaj,
yanlış yapmak büyük zarar üretir. Üç değişmez kural: (1) yalnızca lisanslı/denetimli
kurumlarla çalış, (2) vergiyi net getiri üzerinden planla ve kayıt tut,
(3) mevzuatı yıllık döngüyle (beyan dönemi öncesi) yeniden doğrula.

## Güven Seviyesi
- Kurumsal yapı ve tarihsel vakalar: **%85**
- Kavramsal vergi çerçevesi: **%80**
- Türkiye güncel vergi oranları/istisnaları: **%40 — DOĞRULANMADI (kasıtlı düşük;
  uygulamadan önce resmî teyit şart)**
