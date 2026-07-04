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
| M5 — Risk yönetimi | `05-risk-yonetimi.md` | 🚧 devam | 2026-07-04 |
| M6 — Portföy yönetimi | `06-portfoy-yonetimi.md` | 🚧 devam | 2026-07-04 |
| M7 — Varlık sınıfları | `07-varlik-siniflari.md` | 🚧 devam | 2026-07-04 |
| M8 — Türev ürünler | `08-turev-urunler.md` | 🚧 devam | 2026-07-04 |
| M9 — Davranışsal finans | `09-davranissal-finans.md` | 🚧 devam | 2026-07-04 |
| M10 — Algoritmik işlemler | `10-algoritmik-islemler.md` | 🚧 devam | 2026-07-04 |
| M11 — Küresel piyasalar | `11-kuresel-piyasalar.md` | 🚧 devam | 2026-07-04 |
| M12 — Hukuk ve vergi | `12-hukuk-ve-vergi.md` | 🔜 planlı | — |
| M13 — Güncel durum (kaynaklı anlık görüntü) | `13-guncel-durum-2026.md` | 🔜 planlı | — |
| M14 — Şirket analiz şablonu | `sablonlar/sirket-analiz-sablonu.md` | 🔜 planlı | — |
| M15 — Uzman stratejileri karşılaştırması (Buffett/Dalio/Lynch/Marks…) | `uzmanlar/` | 🔜 planlı | — |
| M16 — Kurum raporları sınıflandırması (SEC/Fed/ECB/TCMB) | `kurum-raporlari/` | 🔜 planlı | — |
| M17 — Kitap özetleri arşivi (artımlı; hedef: geniş yatırım literatürü) | `kitap-ozetleri/` | 🔜 planlı | — |
| M18 — Günlük piyasa takip ajanı (tasarım + kod) | `ajanlar/` | 🔜 planlı | — |

## Oturum Günlüğü

### 2026-07-04 (Oturum 1)
- Metodoloji ve mimari kuruldu; M1–M11 içerikleri yazıldı.
- Güncel makro veriler web'den doğrulandı (Fed %3,50–3,75; ABD TÜFE %4,2 — Mayıs 2026;
  TCMB %37; BTC ~62k $ ve CLARITY Act durumu). Kaynaklar M13'e işlenecek.
- Sıradaki işler: M12, M13, M14; sonra M15–M18.

## Kurallar (özet — ayrıntı README.md)
- Her modül bağımsız commit'lenir; commit mesajı modül adını içerir.
- Her bilgide kaynak + tarih; belirsiz bilgide `[DOĞRULANMADI]` / `[EĞİTİM VERİSİ]` etiketi.
- Tamamlanan dosya yeniden yazılmaz; sadece artımlı güncellenir.
