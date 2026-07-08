# PROJECT_MEMORY.md

Bu dosya `ai-detector-backend` projesinin hafızasıdır. Kod yazmadan önce bu
dosya referans alınmalı; kullanıcı yeni bilgi verdikçe burası güncellenir.

## Çalışma Prensipleri (Tüm Projeler İçin Geçerli)

- Claude bu kullanıcı ile çalışırken CTO, yazılım mimarı ve kıdemli
  geliştirici rolünü üstlenir.
- **Kod yazmadan önce karar verilir ve belgelenir:** Mimari kararlar, veri
  modeli, API sözleşmeleri, teknoloji seçimleri gibi önemli kararlar kod
  yazılmadan önce ilgili projenin `PROJECT_MEMORY.md` dosyasına yazılır.
- **Mimari tutarlılık:** Her yeni özellik, projenin mevcut mimarisine
  (teknoloji yığını, klasör yapısı, kod standartları, kullanılan
  pattern'ler) uygun şekilde tasarlanır. Tutarsız veya mevcut yapıyı
  bozan çözümlerden kaçınılır.
- **Dosya tabanlı proje hafızası:** Her projenin hafızası kendi
  `PROJECT_MEMORY.md` dosyasında tutulur. Kullanıcı yeni bilgi, karar veya
  tercih belirttiğinde ilgili dosya güncellenir; bilgi sohbet geçmişinde
  kaybolmaz.
- Bu prensipler tek bir projeye özel değildir; kullanıcının erişimindeki
  tüm projelerde (repo) geçerlidir ve her reponun kendi
  `PROJECT_MEMORY.md` dosyasında tekrarlanır.

## Genel Bakış

- **Proje adı:** ai-detector-backend
- **Amaç:** Metin ve görsellerin yapay zeka (AI) tarafından üretilip
  üretilmediğini tespit eden bir API servisi.
- **Çalışma şekli:** FastAPI tabanlı tek bir `main.py` dosyası üzerinden
  servis ediliyor.

## Teknoloji Yığını

- **Dil/Framework:** Python, FastAPI, Uvicorn
- **ML/AI:** Hugging Face `transformers`, `torch`, `Pillow`
- **Bağımlılıklar** (`requirements.txt`): fastapi, uvicorn, transformers,
  torch, Pillow, python-multipart

## Proje Yapısı

```
.
├── main.py            # Tüm API endpoint'leri ve model yükleme mantığı
├── requirements.txt   # Python bağımlılıkları
└── README.md
```

## API Endpoint'leri

### `POST /check_text`
- **Girdi:** form alanı `text` (string)
- **Model:** `roberta-base-openai-detector` (text-classification pipeline)
- **Çıktı:** `{ "is_ai": bool, "confidence": float }`
  - `LABEL_1` → AI üretimi (`is_ai: true`)

### `POST /check_image`
- **Girdi:** `file` (UploadFile, görsel)
- **Model:** `nateraw/ai-generated-image-detector`
- **Çıktı:** `{ "label": "AI" | "REAL", "confidence": float }`
  - `label_id == 0` → "AI"

## Mevcut Durum / Notlar

- CORS şu an tüm origin/method/header'lara açık (`allow_origins=["*"]`).
  Production'a geçmeden önce kısıtlanması gerekebilir.
- Modeller her istek için değil, uygulama başlangıcında bir kez yükleniyor
  (`pipeline(...)`, `AutoModelForImageClassification.from_pretrained(...)`
  modül seviyesinde çağrılıyor).
- Henüz test dosyası, CI/CD pipeline'ı veya ortam değişkeni yönetimi yok.

## Kararlar / Tercihler

- **2026-07-04 — Yatırım Araştırma Sistemi bu repoda:** Kullanıcının talep ettiği
  yatırım bilgi tabanı (`investment-research/` dizini) bu repoda barındırılır
  (dal: `claude/investment-research-system-eiikog`). Gerekçe: rentecar-web canlı bir
  ürün; bu repo minimal/genel amaçlı olduğundan dokümantasyon projesi burada tutuldu.
- **Modüler çalışma modeli:** Proje tek oturumda bitirilmez; her modül bağımsız
  tamamlanır ve **ayrı commit** alır. İlerleme `investment-research/PROJECT_STATUS.md`
  dosyasında izlenir; kesinti sonrası oturumlar oradan devam eder ve tamamlanmış
  dosyaları yeniden oluşturmaz.
- **Kaynak/güven disiplini:** Her bilgide kaynak + tarih; doğrulanmamış bilgi
  `[DOĞRULANMADI]` / `[EĞİTİM VERİSİ]` etiketiyle işaretlenir. Yatırım tavsiyesi verilmez.
- **2026-07-08 — Analist rol standardı:** Kullanıcı, şirket/varlık analizlerinde
  kıdemli aracı kurum analisti kalitesinde 12 bölümlük rapor formatı tanımladı →
  `investment-research/sablonlar/analist-rapor-sablonu.md`. Raporlar
  `investment-research/analizler/` altına kaydedilir. Sınırlar korunur: kişiye özel
  al/sat tavsiyesi yok, "kesin/garanti" dili yasak, canlı veri durumu her raporda
  beyan edilir, olumsuz görüş serbest.

## Yapılacaklar / Açık Konular

- Yatırım araştırma sistemi: M15–M18 modülleri (uzman stratejileri, kurum raporları,
  kitap özetleri arşivi, günlük piyasa takip ajanı) — ayrıntı ve öncelik sırası
  `investment-research/PROJECT_STATUS.md` içinde.
- Backend tarafı (AI dedektör): test, CI/CD ve ortam değişkeni yönetimi hâlâ eksik.

## Güncelleme Geçmişi

- İlk oluşturma: mevcut kod tabanı (main.py, requirements.txt) incelenerek
  hazırlandı.
- Genel çalışma prensipleri (CTO/mimar rolü) eklendi.
- 2026-07-04: Yatırım araştırma sistemi kararları ve açık konuları eklendi.
