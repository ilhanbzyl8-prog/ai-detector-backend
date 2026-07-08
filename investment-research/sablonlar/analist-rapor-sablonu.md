# Analist Araştırma Raporu Şablonu (12 Bölüm)

> Kullanıcı tanımlı rol standardı (2026-07-08): kıdemli aracı kurum araştırma
> analisti kalitesinde, veri odaklı, senaryolu rapor. Her şirket/varlık raporu
> bu formatı izler ve `analizler/<TICKER>-<YYYY-AA>.md` olarak kaydedilir.
> M14 (sirket-analiz-sablonu.md) veri toplama aracıdır; bu şablon ise **çıktı**
> formatıdır — M14 doldurulur, bu rapor ondan yazılır.

## Zorunlu Künye (her raporun başına)
- Rapor tarihi / veri kesim tarihi ve saati
- Kullanılan veri kaynakları (KAP/EDGAR, web araması, eğitim verisi) — her sayısal
  verinin yanında kaynak + tarih
- **Canlı veri durumu:** hangi veriler güncel doğrulandı, hangileri eski/teyitsiz
- Sorumluluk reddi: "Bu rapor bilgilendirme amaçlıdır; SPK mevzuatı kapsamında
  yatırım danışmanlığı değildir. Yatırım danışmanlığı, yetkili kuruluşlarca kişiye
  özel sunulur. Her yatırım risk içerir."

## Rapor Bölümleri

### 1. Şirket Özeti
İş modeli, gelir kırılımı, ortaklık yapısı, halka açıklık, endeks üyeliği.

### 2. Finansal Güç Analizi
- Üç tablo özeti (son 4-5 dönem trendiyle): büyüme, marjlar (brüt/FAVÖK/net),
  nakit dönüşümü (FCF/net kâr)
- Oranlar: F/K, PD/DD, FD/FAVÖK, PEG, ROE (DuPont), ROA, Net Borç/FAVÖK,
  faiz karşılama, cari oran — **her oran sektör bağlamıyla yorumlanır, çıplak bırakılmaz**
- TR şirketlerinde: TMS-29 enflasyon muhasebesi baz uyarısı; döviz pozisyonu tablosu
- Kâr kalitesi kırmızı bayrak taraması (M14 §4)

### 3. Teknik Görünüm
Trend/market structure, kilit destek-direnç bölgeleri, 50/200 SMA-EMA konumu,
RSI/MACD durumu, hacim teyidi, varsa belirgin formasyon. **Dürüstlük notu:**
dosya 04 çerçevesinde — teknik bölüm olasılıksal risk/seviye haritasıdır, kehanet
değildir; canlı fiyat verisi yoksa bu bölüm açıkça "veri yok" der veya atlanır.

### 4. Sektör ve Rakip Analizi
Sektör büyüme/marj dinamiği, şirketin pazar payı seyri, 2-4 rakiple çarpan ve
marj karşılaştırma tablosu, rekabet avantajı (hendek) değerlendirmesi.

### 5. Risk Faktörleri
İşe özgü (ilk 3, olasılık × etki), makro (kur/faiz/enflasyon duyarlılığı,
CDS bağlamı), regülasyon, likidite/halka açıklık riskleri. Kur ve faiz
duyarlılığı mümkünse sayısallaştırılır (ör. +%10 kur → net kâr etkisi).

### 6. Güçlü Yönler / 7. Zayıf Yönler
Kanıta bağlı madde listeleri (her madde bir veriye veya kaynağa işaret eder).

### 8-10. Senaryolar (Kısa / Orta / Uzun Vade)
Her senaryo: iyimser-baz-kötümser üçlemesi + her birinin dayandığı 2-3 açık
varsayım + hangi gösterge hangi senaryoyu doğrular/çürütür (izleme tetikleri).
Kısa vade ≈ 0-6 ay (akış/teknik ağırlıklı), orta ≈ 6-24 ay (kârlılık döngüsü),
uzun ≈ 2+ yıl (yapısal tez).

### 11. Adil Değer Değerlendirmesi
- Yöntem seçimi gerekçesi (dosya 03) + DCF senaryo aralığı + emsal çarpan çapraz
  kontrolü + ters DCF ("mevcut fiyat neyi fiyatlıyor?")
- Çıktı **aralıktır**, nokta hedef değil; güvenlik marjı belirtilir
- Hedef fiyat verilecekse "12 aylık, baz senaryo, şu varsayımlarla" diye bağlanır

### 12. Sonuç ve Genel Değerlendirme
Yatırım tezi 3-5 cümlede; karşı-tez (steelman) 2 cümlede; izlenecek 3-5 KPI;
raporun güven seviyesi (%) ve en zayıf varsayımın hangisi olduğu.

## Davranış Kuralları (özet)
- "Kesin yükselir / garanti" dili yasak; her rapor risk uyarısı içerir.
- Haber ile yorum ayrı işaretlenir; varsayımlar açıkça "VARSAYIM:" ile yazılır.
- Canlı veri yoksa/eskiyse tarihiyle beyan edilir; eksik kritik veri kullanıcıdan istenir.
- Olumsuz sonuç ("pahalı", "yatırım yapılabilir bulunmadı") geçerli ve teşvik edilen çıktıdır.
- Bilgi tabanı etiket sistemi (`[DOĞRULANDI]`, `[EĞİTİM VERİSİ]`, `[DOĞRULANMADI]`,
  `[TARTIŞMALI]`) raporlarda da geçerlidir.

*Şablon sürümü: 1.0 (2026-07-08)*
