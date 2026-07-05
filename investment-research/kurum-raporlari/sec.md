# SEC (ABD Menkul Kıymetler Komisyonu) Dosyalamaları

Şirket analizinin (dosya 02) birincil kaynak deposu. Erişim: **EDGAR**
(sec.gov/edgar) — ücretsiz, tam metin aranabilir, API'si var (efts.sec.gov).

## Dosyalama Sınıflandırması

| Form | Sıklık/Tetik | Sinyal | İçerik ve yatırımcı için anlamı |
|---|---|---|---|
| **10-K** | Yıllık | ★ | Denetlenmiş yıllık rapor. Kritik bölümler: Item 1 (iş tanımı), **1A (risk faktörleri — yıldan yıla diff'i oku)**, 7 (MD&A — yönetimin anlatısı), 8 (tablolar + **dipnotlar**). Şirket analizinin başlangıç noktası. |
| **10-Q** | Çeyreklik | ★ | Denetlenmemiş ara dönem; trend takibi ve MD&A güncellemesi. |
| **8-K** | Olay bazlı (4 iş günü içinde) | ★ | "Önemli olay" bildirimi: CEO ayrılığı, denetçi değişikliği (kırmızı bayrak — dosya 02), anlaşma, temerrüt. Gerçek zamanlı izlenmesi gereken form. |
| **DEF 14A (proxy)** | Yıllık (genel kurul öncesi) | ◐ | Yönetici ücretlendirmesi ve **teşvik metrikleri** (neye göre prim alıyorlar — M14 şablonu §5), yönetim kurulu bağımsızlığı, ilişkili taraf işlemleri. |
| **Form 4** | İçeriden işlemde 2 iş günü | ◐ | İçeridekilerin alım/satımı. Sinyal asimetrik: **alımlar bilgilendirici** (tek neden: yükseliş beklentisi), satışlar çok nedenli (vergi, çeşitlendirme). Küme halinde alım en güçlü sinyal (literatür: insider alımlarında anormal getiri — Lakonishok & Lee 2001). |
| **13F** | Çeyreklik (+45 gün gecikme) | ◐ | Kurumsal yöneticilerin uzun pozisyonları ("guru takibi"). Tuzaklar: 45 gün bayat, short'lar görünmez, opsiyonlar kısmi. Kopyalama stratejisinin ampirik sonucu karışık. `[TARTIŞMALI]` |
| **13D / 13G** | %5+ pay alımında | ★ | Aktivist niyet (13D) vs. pasif pay (13G). 13D dosyalaması hissede olay başlatır. |
| **S-1** | Halka arz öncesi | ◐ | IPO izahnamesi; iş modeli + risklerin en dürüst yazıldığı yer (dava riski nedeniyle). Kilitleme (lock-up) tarihleri arz sonrası satış baskısı takvimi. |
| **20-F / 6-K** | Yıllık / ara (yabancı ihraççı) | ◐ | ABD'de kote yabancı şirketler (Türk ADR'leri dahil). |
| **NT 10-K/10-Q** | Gecikme bildirimi | ★ | Raporunu zamanında veremeyen şirket — güçlü kırmızı bayrak. |

## Okuma Pratiği
1. Yeni şirkette sıra: son 10-K (tamamı) → son 2 çeyrek 10-Q → son 12 ay 8-K'ları →
   proxy (teşvikler) → Form 4 örüntüsü.
2. **Diff okuma tekniği:** Risk faktörleri ve MD&A'da yıldan yıla eklenen/çıkan
   cümleler, yeni bilginin kendisidir (avukatlar sebepsiz cümle eklemez).
3. Dipnot öncelikleri: gelir tanıma politikası, borç vadeleri, kiralamalar,
   taahhütler/davalar, segment raporu.
4. EDGAR tam metin araması ile tema taraması (ör. tüm 10-K'larda "going concern").

## Güven Seviyesi
Form işlevleri: **%95** (mevzuat istikrarlı). Süre/eşik detayları güncellenebilir `[EĞİTİM VERİSİ]`.
