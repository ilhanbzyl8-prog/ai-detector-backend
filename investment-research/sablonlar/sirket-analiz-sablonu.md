# Şirket Analiz Şablonu

> Her şirket analizi bu şablonu doldurur ve `analizler/<TICKER>-<YYYY-AA>.md`
> olarak kaydedilir. Boş bırakılan alan "bilinmiyor" demektir — tahmin uydurulmaz.
> Her veri noktasına kaynak + tarih yazılır. İlgili yöntem detayları:
> dosya 02 (şirket analizi), dosya 03 (değerleme), dosya 05 (risk).

---

## 0. Kimlik ve Analiz Künyesi
- **Şirket / Ticker / Borsa:**
- **Analiz tarihi / Analist:**
- **Güncel fiyat / Piyasa değeri / Halka açıklık:**
- **Veri kaynakları:** (10-K/faaliyet raporu linki, KAP/EDGAR, veri sağlayıcı)
- **Analizin güven seviyesi (%):** (tamamlanınca doldur)

## 1. İş Modeli (para nasıl kazanılıyor?)
- Ürün/hizmet ve gelir kırılımı (segment, coğrafya, %):
- Müşteri profili ve yoğunlaşması (ilk 5 müşteri payı):
- Fiyatlama gücü kanıtı (geçmiş zam geçirgenliği):
- Döngüsellik: (döngüsel / savunmacı / yapısal büyüme)
- **Tek cümle testi:** "Bu şirket ____ satarak, ____ müşterisinden, ____ nedeniyle
  rakiplerinden daha kârlı para kazanır." (Doldurulamıyorsa analiz durur.)

## 2. Rekabet ve Hendek (Moat)
- Hendek türü: [ ] maliyet avantajı [ ] ağ etkisi [ ] geçiş maliyeti
  [ ] marka/patent [ ] ölçek [ ] regülasyon [ ] YOK
- Hendek kanıtı (sayısal): 5–10 yıl ROIC > WACC mi? Pazar payı seyri?
- Ana rakipler ve göreli konum:
- Bozulma (disruption) tehdidi:

## 3. Finansal Sağlık (son 5–10 yıl)
| Metrik | Y-4 | Y-3 | Y-2 | Y-1 | Son | Eğilim |
|---|---|---|---|---|---|---|
| Hasılat büyümesi % | | | | | | |
| Brüt marj % | | | | | | |
| EBIT marjı % | | | | | | |
| Net marj % | | | | | | |
| ROIC % | | | | | | |
| ROE % (DuPont notu) | | | | | | |
| FCF marjı % | | | | | | |
| FCF / Net kâr | | | | | | |
| Net borç / FAVÖK | | | | | | |
| Faiz karşılama (EBIT/faiz) | | | | | | |
| Hisse sayısı (seyreltilmiş) | | | | | | |

- Çalışma sermayesi döngüsü (alacak/stok/borç günleri) notu:
- Borç vade yapısı ve para birimi uyumu:
- Bilanço dışı yükümlülükler (kira, dava, taahhüt, emeklilik):

## 4. Kâr Kalitesi ve Kırmızı Bayrak Taraması
- [ ] İşletme nakit akışı ile net kâr uyumlu (FCF/NI > ~0,7 ortalama)
- [ ] Alacaklar hasılattan hızlı büyümüyor
- [ ] "Tek seferlik" giderler gerçekten tek seferlik
- [ ] Gelir tanıma politikası agresif değil
- [ ] Denetçi görüşü olumlu; denetçi değişikliği yok
- [ ] İçeriden satış dalgası yok; SBC makul
- [ ] İlişkili taraf işlemleri sınırlı
- Beneish M-Score / Altman Z-Score (varsa):
- **Herhangi bir kutu boşsa nedeni yazılır.**

## 5. Yönetim ve Sermaye Tahsisi
- CEO/yönetim geçmişi ve pay sahipliği:
- Son 5 yıl sermaye tahsisi: capex / temettü / geri alım / satın alma (TL veya $ kırılımı):
- Satın almaların geçmiş getirisi (değer yarattı mı?):
- Hissedar mektubu dürüstlük testi (kötü yılları kabulleniyor mu?):
- Teşvik yapısı neye bağlı (EPS mi ROIC mi hisse fiyatı mı?):

## 6. Büyüme ve Senaryolar (5–10 yıl)
| Varsayım | Kötü | Baz | İyi |
|---|---|---|---|
| Hasılat CAGR % | | | |
| Uç yıl EBIT marjı % | | | |
| Yeniden yatırım ihtiyacı | | | |
| Uç yıl FCF | | | |
- Baz senaryonun dayandığı 2–3 kritik varsayım (ve nasıl yanlışlanır):

## 7. Değerleme (dosya 03 yöntemleri)
- **DCF:** İskonto oranı (gerekçeli), terminal büyüme, senaryo başına değer:
  - Kötü: ___ | Baz: ___ | İyi: ___
- **Çarpanlar:** P/E, EV/EBIT, EV/FCF — tarihsel kendi aralığı + emsaller tablosu:
- **Çapraz kontrol:** DCF ile çarpan sonucu tutarlı mı? Değilse neden?
- **İçsel değer aralığı:** ___ – ___
- **Güvenlik marjı @ güncel fiyat:** %___
- **Ters DCF:** Bugünkü fiyat hangi büyüme/marj varsayımını fiyatlıyor? Makul mü?

## 8. Risk Haritası
- İşe özgü ilk 3 risk (olasılık × etki):
- Makro duyarlılık (faiz, kur, emtia, regülasyon):
- Tez neyle çürür? (**önceden yazılmış çıkış kriteri** — dosya 05/09)
- Premortem: "3 yıl sonra bu yatırım battı; en olası neden ___ idi."

## 9. Karar Kaydı (davranışsal koruma — dosya 09)
- Tez (3 cümle):
- Karşı tez (en güçlü haliyle — steelman):
- Beklenen tutma süresi ve gözden geçirme tarihi:
- Pozisyon boyutu ve gerekçesi (dosya 05 kurallarına referans):
- Karar anındaki duygu notu (FOMO/korku kontrolü):

## 10. İzleme Planı
- Çeyreklik takip edilecek 3–5 KPI:
- Tez sağlığı eşikleri (KPI hangi değeri görürse alarm):
- Sonraki gözden geçirme tarihi:

---
*Şablon sürümü: 1.0 (2026-07-04). Değişiklikler commit geçmişinde izlenir.*
