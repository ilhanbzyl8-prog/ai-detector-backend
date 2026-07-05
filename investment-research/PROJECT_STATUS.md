# PROJECT_STATUS — Yatırım Araştırma Sistemi

> **Devam prosedürü:** Kesinti (API hatası, bağlantı kopması, token sınırı) sonrası
> yeni oturum ÖNCE bu dosyayı okur. `✅ tamam` işaretli dosyalar YENİDEN OLUŞTURULMAZ;
> yalnızca `🚧 devam` ve `🔜 planlı` modüller üzerinde çalışılır. Güncel veri içeren
> dosyalar (özellikle 13) her oturumda revize edilebilir — bu "yeniden oluşturma" değil
> "güncelleme"dir ve dosya içindeki tarih + commit mesajıyla izlenir.

## Modül Durumu

| Modül | Dosya | Durum | Son güncelleme |
|---|---|---|---|
| M0 — Mimari, klasör yapısı, metodoloji | `README.md` | ✅ tamam | 2026-07-04 |
| M1 — Makroekonomi | `01-makroekonomi.md` | ✅ tamam | 2026-07-04 |
| M2 — Şirket analizi | `02-sirket-analizi.md` | ✅ tamam | 2026-07-04 |
| M3 — Değerleme | `03-degerleme.md` | ✅ tamam | 2026-07-04 |
| M4 — Teknik analiz | `04-teknik-analiz.md` | ✅ tamam | 2026-07-04 |
| M5 — Risk yönetimi | `05-risk-yonetimi.md` | ✅ tamam | 2026-07-04 |
| M6 — Portföy yönetimi | `06-portfoy-yonetimi.md` | ✅ tamam | 2026-07-04 |
| M7 — Varlık sınıfları | `07-varlik-siniflari.md` | ✅ tamam | 2026-07-04 |
| M8 — Türev ürünler | `08-turev-urunler.md` | ✅ tamam | 2026-07-04 |
| M9 — Davranışsal finans | `09-davranissal-finans.md` | ✅ tamam | 2026-07-04 |
| M10 — Algoritmik işlemler | `10-algoritmik-islemler.md` | ✅ tamam | 2026-07-04 |
| M11 — Küresel piyasalar | `11-kuresel-piyasalar.md` | ✅ tamam | 2026-07-04 |
| M12 — Hukuk ve vergi | `12-hukuk-ve-vergi.md` | ✅ tamam | 2026-07-04 |
| M13 — Güncel durum (kaynaklı anlık görüntü) | `13-guncel-durum-2026.md` | ✅ tamam | 2026-07-04 |
| M14 — Şirket analiz şablonu | `sablonlar/sirket-analiz-sablonu.md` | ✅ tamam | 2026-07-04 |
| M15 — Uzman stratejileri karşılaştırması (9 yatırımcı + matris) | `uzmanlar/` | ✅ tamam | 2026-07-04 |
| M16 — Kurum raporları sınıflandırması (SEC/Fed/ECB/TCMB) | `kurum-raporlari/` | 🔜 planlı | — |
| M17 — Kitap özetleri arşivi (artımlı; hedef: geniş yatırım literatürü) | `kitap-ozetleri/` | 🔜 planlı | — |
| M18 — Günlük piyasa takip ajanı (tasarım + kod) | `ajanlar/` | 🚧 v1 hazır — canlı doğrulama bekliyor | 2026-07-04 |

## Oturum Günlüğü

### 2026-07-04 (Oturum 2) — M18 v1
- Kullanıcı talebiyle M18 öne alındı (M15 sıradaki ilk iş olarak duruyor).
- Tasarım dokümanı yazıldı: `ajanlar/00-tasarim-gunluk-piyasa-takip.md`.
- Kod tamamlandı: `ajanlar/piyasa-takip/` (agent.py + config.py + test_agent.py +
  README). Yalnızca standart kütüphane; Stooq/CoinGecko/FRED anahtarsız kaynaklar.
- **13/13 birim testi geçti** (ağsız). Ağ kapalıyken davranış doğrulandı: gösterge
  başına "VERİ YOK + neden", rapor yine üretiliyor.
- **Açık iş (canlı doğrulama):** sandbox dış ağı kapalı (proxy 403) → ilk gerçek
  çalıştırma kullanıcı ortamında; `^xu100` ve `cb.f` sembol teyidi; ardından durum
  ✅'ye çekilecek. GitHub Actions cron'u kullanıcı onayı bekliyor.

### 2026-07-04 (Oturum 1) — TAMAMLANDI
- Metodoloji ve mimari kuruldu; **M0–M14 tamamlandı**, her modül ayrı commit ile kaydedildi.
- Güncel makro veriler web'den doğrulandı ve kaynaklarıyla M13'e işlendi
  (Fed %3,50–3,75; ABD TÜFE %4,2 — Mayıs 2026; TCMB %37; BTC ~62k $; CLARITY Act durumu).
- **Sonraki oturum görevleri (öncelik sırasıyla):**
  1. M15 — `uzmanlar/`: Buffett, Dalio, Lynch, Marks, Graham, Munger, Soros, Wood,
     Damodaran strateji karşılaştırmaları (yatırımcı başına 1 dosya + karşılaştırma matrisi).
  2. M16 — `kurum-raporlari/`: SEC/Fed/ECB/TCMB rapor türleri sınıflandırması
     (hangi rapor, hangi sıklıkta, yatırımcı için hangi sinyal).
  3. M17 — `kitap-ozetleri/`: artımlı kitap analiz arşivi (önce çekirdek ~20 klasik;
     her kitap: tez, kanıt, eleştiri, güncelliği). "1000+ kitap" hedefi artımlıdır —
     oturum başına parti parti büyütülür, tek seferde denenmez.
  4. M18 — `ajanlar/`: günlük piyasa takip ajanı — önce tasarım dokümanı
     (veri kaynakları, tetikleme, çıktı formatı, M13'ü otomatik güncelleme akışı), sonra kod.
  5. M13 revizyonu — açık konular: TR TÜFE, BIST, ECB/BoJ, petrol/altın, S&P 500/CAPE.

### 2026-07-04 (Oturum 3) — M15 tamam
- `uzmanlar/` tamamlandı: Graham, Buffett, Munger, Lynch, Marks, Dalio, Soros,
  Wood, Damodaran (ortak şablon: felsefe → mekanik → belgelenmiş performans →
  başarısızlıklar → eleştiriler → çelişkiler → taşınabilir dersler) +
  `karsilastirma.md` (matris, eksen haritası, uzlaşı/çelişki tabloları, sentez).
- Kayıt kalitesi ilkesi uygulandı: halka açık kayıtlar (Berkshire, Magellan, ARKK)
  ile denetimsiz hedge fon anlatıları (Quantum, Bridgewater, Oaktree) ayrı etiketlendi.
- **Kalan modüller:** M16 (kurum raporları), M17 (kitap özetleri — artımlı),
  M18 canlı doğrulama, M13 revizyonu (açık veri konuları).

## Kurallar (özet — ayrıntı README.md)
- Her modül bağımsız commit'lenir; commit mesajı modül adını içerir.
- Her bilgide kaynak + tarih; belirsiz bilgide `[DOĞRULANMADI]` / `[EĞİTİM VERİSİ]` etiketi.
- Tamamlanan dosya yeniden yazılmaz; sadece artımlı güncellenir.
