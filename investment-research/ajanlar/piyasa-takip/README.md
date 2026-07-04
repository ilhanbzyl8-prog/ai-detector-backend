# Günlük Piyasa Takip Ajanı (v1)

Tasarım: [`../00-tasarim-gunluk-piyasa-takip.md`](../00-tasarim-gunluk-piyasa-takip.md)

## Kullanım
```bash
python3 agent.py            # raporu raporlar/YYYY-MM-DD.md dosyasına yazar
python3 agent.py --dry-run  # dosya yazmadan stdout'a basar
python3 -m unittest test_agent -v   # ağsız birim testleri (13 test)
```
- Bağımlılık yok (yalnızca Python 3.10+ standart kütüphane).
- Kaynaklar: Stooq (endeks/kur/emtia), CoinGecko (kripto), FRED (faiz) — anahtarsız.
- Kaynak hatasında değer uydurulmaz; raporda `VERİ YOK` + neden görünür.
- `raporlar/latest.json` son başarılı değerleri tutar; günlük Δ ve eşik uyarıları
  (⚠️ DİKKAT bölümü) buradan hesaplanır.

## Durum / bilinen sınırlar
- **Canlı uçtan uca test henüz yapılmadı:** geliştirme sandbox'ında dış ağ kapalı
  (proxy 403; 2026-07-04 test edildi). Ağsız 13 birim testi geçiyor; hata yolunun
  tamamı (`--dry-run`, ağ kapalıyken) doğrulandı.
- İlk canlı çalıştırmada `^xu100` (BIST 100) ve `cb.f` (Brent) sembollerini kontrol
  edin; `N/D` dönerse `config.py` içinde düzeltin.
- Zamanlama (GitHub Actions cron) v1'e bilinçli dahil edilmedi — repoya CI eklemek
  kullanıcı onayı bekliyor (tasarım §6).

*Bu araç bilgi toplar; yatırım tavsiyesi üretmez.*
