# TCMB ve Türkiye Veri/Rapor Ekosistemi

TL varlıkları (BIST, tahvil, kur) için birincil kaynak seti. Yüksek enflasyon
ortamında veri okuma disiplini (nominal/reel ayrımı) diğer ülkelerden daha kritiktir.

## TCMB (tcmb.gov.tr)

| Yayın | Sıklık | Sinyal | İçerik ve yatırımcı için anlamı |
|---|---|---|---|
| **PPK kararı + kısa metin** | Yılda 8 | ★ | Politika faizi (1 hafta repo) + koridor. Metindeki tek cümle değişimi (ör. "sıkı duruş sürdürülecektir" → "gerektiğinde…") döngü sinyalidir. Haziran 2026 örneği: enerji şoku gerekçesiyle indirim döngüsünün durdurulması (dosya 13). |
| **PPK özeti** | Karardan ~5 iş günü sonra | ◐ | Kararın gerekçe ayrıntısı. |
| **Enflasyon Raporu** | Çeyreklik | ★ | Enflasyon tahmin patikası + çıktı açığı analizi. Tahmin revizyonları (yukarı/aşağı) politika patikasının öncüsüdür; basın toplantısı soru-cevabı önemli. |
| **Piyasa Katılımcıları Anketi** | Aylık | ★ | Piyasanın enflasyon/kur/faiz beklentileri; **beklenti çıpası bozulması** (anket ↑ hedef sabitken) rejim sinyalidir. |
| **Finansal İstikrar Raporu** | Yılda 2 | ◐ | Hane/şirket/banka bilanço riskleri, dolarizasyon. |
| **EVDS (veri sistemi)** | Sürekli | ★ | Rezervler (brüt/net, swap hariç net — kritik ayrım), KKM bakiyesi, menkul kıymet istatistikleri, yabancı payı, kredi büyümesi, konut fiyat endeksi. API'si var (anahtar gerektirir — piyasa ajanı v2 hedefi, M18). |
| **Haftalık menkul kıymet istatistikleri** | Haftalık | ◐ | Yabancı giriş/çıkışı — sermaye akışı nabzı (dosya 11 dolar döngüsü). |

## TÜİK (tuik.gov.tr)

| Yayın | Sıklık | Sinyal | Not |
|---|---|---|---|
| **TÜFE/Yİ-ÜFE** | Aylık (ayın ~3'ü) | ★ | Aylık ve yıllık; ÜFE-TÜFE makası marj baskısı göstergesi. Bağımsız doğrulama tartışması: ENAG gibi alternatif ölçümlerle fark kamuoyu tartışmasıdır — analizde resmî seri kullanılır, tartışma not edilir. `[TARTIŞMALI]` |
| GSYH | Çeyreklik | ◐ | Revizyonlara açık. |
| İşgücü, sanayi üretimi, perakende | Aylık | ◐ | Büyüme nabzı. |

## KAP (kap.org.tr) — şirket dosyalamalarının EDGAR karşılığı

| Bildirim | Sinyal | Not |
|---|---|---|
| **Finansal tablolar + dipnotlar** | ★ | Çeyreklik; TMS-29 **enflasyon muhasebesi** düzeltmeli — yıllar arası karşılaştırmada düzeltme bazı kontrol edilmeli (dosya 11 Türkiye bölümü). |
| **Özel durum açıklamaları (ÖDA)** | ★ | 8-K karşılığı; gerçek zamanlı izlenir. |
| Faaliyet raporları, kurumsal yönetim | ◐ | Yönetim kalitesi analizi (M14 §5). |
| Pay geri alım bildirimleri | ◐ | BIST'te yaygınlaşan sinyal. |

## SPK (spk.gov.tr)
- **Haftalık bülten** ★: işlem yasakları, cezalar, onaylanan ihraçlar — manipülasyon
  vakalarının resmî kaydı (dosya 12).
- Aracı kurum/fon istatistikleri ◐: TEFAS (tefas.gov.tr) fon karşılaştırma —
  Türk fonlarının getiri/ücret şeffaflığı için ana araç ★.

## Okuma Pratiği (Türkiye'ye özgü)
1. **Her seriyi üç bazda oku:** nominal TL / USD / reel (TÜFE düzeltmeli) —
   yalnız nominal okuma sistematik yanılgı üretir (dosya 11).
2. Rezerv verisinde "swap hariç net rezerv"i ayrıca hesapla/izle; manşet brüt rezerv
   yanıltıcı olabilir.
3. Beklenti anketi + kur + risk primi (CDS) üçlüsü, politika sürdürülebilirliğinin
   hızlı testi.
4. KAP dipnotlarında kur riski tablosu (döviz pozisyonu) Türk şirket analizinin
   zorunlu adımı (2018 dersi — dosya 02).

## Güven Seviyesi
Kurum/rapor işlevleri: **%85**. Yayın takvimleri ve KKM gibi araçların güncel durumu
değişkendir `[EĞİTİM VERİSİ — ilk kullanımda doğrula]`.
