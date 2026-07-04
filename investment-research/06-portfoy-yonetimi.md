# 06 — Portföy Yönetimi

## Konunun Özeti
Portföy yönetimi, tekil varlık seçiminin ötesinde, varlıkların **birlikte** nasıl
davrandığını yöneterek hedef getiri/risk profiline ulaşma disiplinidir. Ampirik bulgu:
uzun vadeli getiri oynaklığının büyük bölümü varlık dağılımı (asset allocation)
politikasıyla açıklanır (Brinson vd. 1986/1991 — sıkça "getirinin %90'ı" diye yanlış
aktarılır; bulgu getiri düzeyi değil, **oynaklığın** açıklanmasıdır). `[TARTIŞMALI — yorumu üzerinde literatür tartışması var]`

## Temel Kavramlar

### Modern Portföy Teorisi (MPT)
- Markowitz (1952): Yatırımcı, beklenen getiri–varyans düzleminde optimize eder;
  **etkin sınır** = her risk düzeyinde maksimum beklenen getiri veren portföyler.
- CAPM (Sharpe 1964): Denge modelinde tek fiyatlanan risk piyasa betasıdır.
  Ampirik olarak zayıflamıştır (düşük beta anomalisi, faktörler) ama dil ve çerçeve
  olarak hâlâ temeldir.
- Pratik zaaf: Optimizasyon, girdi hatalarını büyütür ("error maximizer" —
  Michaud 1989); beklenen getiri tahminindeki ufak oynama köşe çözümler üretir.
  Çözümler: kısıtlar, Black-Litterman (1992), yeniden örnekleme, eşit ağırlık.
  DeMiguel vd. (2009): naif 1/N portföyü, örneklem dışında çoğu optimizasyonu yendi.

### Asset Allocation
- **Stratejik (SAA):** Uzun vadeli politika ağırlıkları (ör. 60/40).
- **Taktik (TAA):** Değerleme/momentum sinyalleriyle kısa vadeli sapmalar — katma
  değeri ampirik olarak tartışmalı, maliyeti kesin.
- Yaşam döngüsü yaklaşımı: insan sermayesi gençken tahvil benzeri kabul edilip
  finansal portföyde hisse ağırlığı yüksek tutulur; hedef tarih fonlarının mantığı.

### Rebalancing (Yeniden Dengeleme)
- Hedef ağırlıklara dönüş: takvimsel (yıllık/çeyreklik) veya eşik bazlı (±%5 sapma).
- Etkisi: riski hedefte tutar (ana işlev) + ortalamaya dönen varlıklar arasında küçük
  bir "rebalancing bonusu" yaratabilir; güçlü trendli dönemlerde ise kazananı erken
  kırptığı için maliyetli olabilir. Getiri etkisi ikincil, **risk kontrolü birincil** gerekçedir.

### Hedge Stratejileri
- Araçlar: endeks put'u (pahalı ama dışbükey), put spread/collar (ucuz, sınırlı),
  uzun volatilite, trend takibi (kriz alfası), nakit/kısa vadeli tahvil, altın,
  para birimi hedge'i, short pozisyonlar.
- Temel gerilim: sürekli kuyruk sigortası taşımak pahalıdır (varyans risk primi
  sigorta satıcısına akar); taşımamak kuyrukta yıkıcıdır. Kurumsal pratik: küçük ve
  sürekli dışbükey koruma + rejim sinyaliyle ayarlama. `[TARTIŞMALI — Universa/Taleb
  sürekli tail hedge savunur; AQR (Asness) maliyetinin uzun vadede trend takibinden
  kötü olduğunu savunur]`

### Faktör Yatırımı (Smart Beta)
- **Value:** Ucuz (P/B, P/E, EV/EBIT düşük) hisselerin uzun vadeli primi
  (Fama-French 1992). 2010–2020 arası tarihinin en kötü on yılını yaşadı; 2021–22'de
  toparlandı. Ölümü ilan edildi, geri döndü — faktör dönemselliği dersi.
- **Momentum:** 3–12 ay kazananları al (Jegadeesh & Titman 1993). En güçlü ve en
  yaygın belgelenmiş anomali; zaafı: keskin çöküşler ("momentum crash" 2009).
- **Quality:** Yüksek kârlılık, düşük borç, istikrar (Novy-Marx 2013; QMJ —
  Asness vd.). Ayı piyasasında görece dayanıklı.
- **Size (Small Cap):** Küçük şirket primi (Banz 1981) — sonraki araştırmalarda en
  zayıf faktör; birçok araştırmacı bağımsız prim olarak sorgular (kalite ile
  birleşince anlamlı: "küçük ve sağlam"). `[TARTIŞMALI]`
- **Low Volatility:** Düşük oynaklıklı hisselerin riske göre yüksek getirisi —
  CAPM'e doğrudan aykırı anomali.
- Uygulama riskleri: kalabalıklaşma, faktör tanım farkları, yüksek rotasyon maliyeti,
  10+ yıl sürebilen düşük performans dönemleri (kariyer riski).

### Growth vs. Value vs. Blend
- "Growth" akademik faktör değil stil etiketidir (yüksek büyüme beklentili, pahalı
  çarpanlı hisseler). Uzun dönem verisi value lehinedir ama rejim bağımlılığı yüksektir:
  düşük faiz + teknolojik platform ekonomisi (2010'lar) growth'u; enflasyon/faiz
  şokları (1970'ler, 2022) value'yu destekledi.

## Nasıl Çalışır? (Portföy Kurulum Süreci)
1. **Amaç ve kısıtlar:** Ufuk, likidite ihtiyacı, drawdown toleransı, vergi durumu.
2. **Politika portföyü (SAA):** Varlık sınıfı ağırlıkları + kabul edilebilir bantlar.
3. **Uygulama araçları:** Düşük maliyetli endeks/ETF çekirdek + (istenirse) uydu
   aktif/faktör pozisyonlar ("core-satellite").
4. **Rebalancing kuralı** yazılı olarak belirlenir (duygusal karar dışı bırakılır).
5. **İzleme:** Performans, politikaya göre sapma, maliyet/vergi verimliliği;
   kıyas ölçütü (benchmark) baştan tanımlanır.

## Avantajlar
- Tekil hata riskini sistemikleştirir: tek hissede yanılmak portföyü batırmaz.
- Davranışsal koruma: yazılı politika, panik/coşku kararlarını frenler.
- Maliyet ve vergi verimliliği planlanabilir hale gelir.

## Dezavantajlar
- Ortalama sonuç garantisi: iyi çeşitlendirilmiş portföy hiçbir zaman "en iyi" varlık
  kadar kazandırmaz (çeşitlendirmenin duygusal maliyeti: her yıl bir şeyden pişmanlık).
- Model girdilerinin (beklenen getiri, korelasyon) kırılganlığı.
- Aşırı mühendislik riski: karmaşıklık maliyet ve hata alanı ekler.

## Riskler
- **Rejim değişimi:** Hisse-tahvil korelasyonu 2000–2020 negatifti (60/40'ın altın çağı);
  enflasyonist rejimde pozitife döner (1970'ler, 2022) ve klasik dengeleme bozulur.
- **Kalabalık strateji riski:** Aynı faktöre/paritelere yığılma (Ağustos 2007 quant
  quake; risk-parite fonlarının 2020 Mart'ta eşzamanlı deleveraging'i).
- **Likidite uyumsuzluğu:** Günlük likidite vaat eden fonların likit olmayan varlık
  tutması (Woodford fonu 2019, İngiltere).

## Gerçek Örnekler
- **Yale Modeli (David Swensen):** Alternatif ağırlıklı (PE, hedge fon, gerçek varlık)
  bağış fonu; on yıllarca piyasa üstü getiri. Swensen'in kendi uyarısı: bireysel
  yatırımcı bu modeli kopyalayamaz (erişim + kadro + ufuk farkı) → bireye önerisi
  düşük maliyetli endeks çeşitlendirmesi (*Unconventional Success*).
- **Norveç Varlık Fonu (NBIM):** ~%70 hisse, kural bazlı, şeffaf, faktör eğilimli;
  "sıkıcı ama ölçeklenebilir" kurumsal şablon.
- **Bridgewater All Weather:** Büyüme/enflasyon rejimlerine göre risk paritesi;
  2022'de enflasyon şokunda beklenen korumayı tam veremedi — her modelin rejim
  varsayımı vardır. `[EĞİTİM VERİSİ]`
- **60/40'ın yüzyılı:** ABD verisiyle uzun dönem reel ~%5 civarı bileşik getiri;
  1931, 1974, 2008, 2022 gibi ağır yıllar dahil. Basitliğin gücüne kanıt. `[EĞİTİM VERİSİ — yaklaşık]`

## Tarihsel Olaylar
- 1952 Markowitz; 1964 CAPM; 1973 ilk endeks fonu fikirleri, 1976 Vanguard First
  Index Investment Trust (Bogle).
- 1992 Fama-French üç faktör; 2013 Fama'ya Nobel (Shiller ile birlikte — ironik ikili).
- 2008: korelasyon yakınsaması; "çeşitlendirme öldü" tartışması.
- 2010'lar: pasif yatırımın yükselişi (ABD hisse fonlarında pasif payı %50'yi aştı);
  "pasif balonu" tartışması (fiyat keşfi zayıflıyor mu? — kanıt karışık). `[TARTIŞMALI]`

## En Yaygın Hatalar
1. Politika yazmadan portföy kurmak (her karar ad-hoc ve duygusal olur).
2. Performans kovalamak: geçen yılın kazanan fonuna/temasına girmek (Morningstar
   verisi: yatırımcı getirisi fon getirisinin sistematik altında — "behavior gap",
   yıllık ~%1–2). 
3. Ev ülkesi yanlılığı (home bias): portföyün tamamının tek ülke/tek para riskine bağlanması.
4. Sahte çeşitlendirme: 15 farklı ama hepsi aynı faktöre (ör. ABD büyüme hissesi) yüklü pozisyon.
5. Rebalancing'i boğa piyasasında unutmak, ayıda yapamamak.
6. Maliyeti küçümsemek: yıllık %1 ek ücret, 30 yılda son servetin ~%25'ini götürür.
7. Kıyas ölçütü değiştirerek başarıyı yeniden tanımlamak.

## Uzman Görüşleri
- **Bogle:** Maliyet hipotezi — "getirinin en güvenilir belirleyicisi maliyettir";
  toplam piyasa endeksi + tut.
- **Swensen:** Kurumlar için alternatifler, bireyler için endeks; ikisini karıştırma.
- **Dalio:** Rejim dengeli risk paritesi; "bilmediğine karşı çeşitlendir".
- **Buffett:** Çeşitlendirme "cehalete karşı koruma"dır; bilenin konsantre olması
  gerektiğini savunur. Vasiyet portföyü: %90 S&P 500 + %10 kısa vadeli tahvil.
- **Asness (AQR):** Faktörler gerçek ama mükemmel değil; disiplin ve düşük maliyetle
  uzun vade taşınmalı; zamanlamaya kalkışma ("sin a little" en fazla).
- **Çelişki haritası:** Konsantrasyon (Buffett/Munger) vs. geniş çeşitlendirme
  (Bogle/Dalio); pasif (Bogle) vs. faktör-aktif (Asness) vs. tam aktif (Lynch).
  Ortak payda: maliyet düşük, disiplin yüksek, ufuk uzun.

## Akademik Çalışmalar
- Markowitz (1952); Sharpe (1964); Tobin (1958, ayrım teoremi).
- Brinson, Hood & Beebower (1986), politika ağırlıklarının açıklayıcılığı, *FAJ*.
- Fama & French (1992, 1993, 2015 beş faktör), *JF/JFE*.
- Jegadeesh & Titman (1993) momentum; Carhart (1997) dört faktör.
- DeMiguel, Garlappi & Uppal (2009), "1/N", *RFS*.
- Black & Litterman (1992), görüş-denge birleşimi, *FAJ*.
- Hou, Xue & Zhang (2015) q-faktör; Harvey vd. (2016) "factor zoo" ve p-hacking eleştirisi.

## Kaynaklar
- Bogle, *The Little Book of Common Sense Investing*; Swensen, *Unconventional Success*
- Ilmanen, *Expected Returns* (kurumsal referans); Bernstein, *The Intelligent Asset Allocator*
- AQR ve Research Affiliates makale arşivleri; MSCI/S&P faktör endeks metodolojileri
- Credit Suisse/UBS Global Investment Returns Yearbook (Dimson-Marsh-Staunton uzun dönem verisi)

## Güncel Gelişmeler
- Enflasyonun 2026'da yeniden yükselmesi (ABD TÜFE %4,2 — Mayıs 2026 `[DOĞRULANDI]`),
  hisse-tahvil korelasyonu ve 60/40'ın koruyuculuğu tartışmasını canlı tutuyor;
  emtia/reel varlık tahsisi yeniden gündemde. `[Yorum — analitik çıkarım]`

## Sonuç
Portföy yönetiminin %80'i üç kararda biter: (1) maliyeti düşük tut, (2) yazılı bir
dağılım politikası kur ve rebalance et, (3) davranışsal hatalardan koru (dosya 09).
Faktörler ve taktik oyunlar bu çekirdeğin üstüne, ancak disiplinle eklenmelidir.
Mükemmel portföy değil, **sahibinin krizde bile taşıyabileceği** portföy kazanır.

## Güven Seviyesi
- MPT/faktör literatürü özetleri: **%90**
- Faktör primlerinin geleceğe taşınabilirliği: **%60 — [TARTIŞMALI]**
- Vaka örnekleri: **%85**
- Pratik öneri çerçevesi: **%80**
