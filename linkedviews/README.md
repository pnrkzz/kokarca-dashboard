# Linked Views — Halyomorpha halys

15 Ekim 2026 tarihinde paylaşılan React/Vite ve FastAPI kaynak kodu.

## Durum

Frontend üretim derlemesi başarılıdır. Canlı demo henüz yayınlanmamıştır. Backend yerel veri dosyaları ve animasyon için MongoDB gerektirir. GitHub Pages yalnızca frontend dosyalarını sunar; bu Python API uygulamasını çalıştırmaz.

## Çalıştırma

Frontend: `cd frontend`, `npm ci`, `npm run dev`.

Backend: `cd backend`, `pip install -r requirements.txt`, ardından ortam değişkenlerini tanımlayıp `uvicorn main:app --port 8001` çalıştırın. `.env.example` örnektir; değişkenler çalıştırma ortamına aktarılmalıdır.

## Veri gereksinimleri

- Doğrulanmış ana olay CSV ve il sınır GeoJSON dosyası
- `data_processed/STC_IKLIM/01_STC/final_locked/visualization_data/FINAL_hazelnut_province_year_cube.csv`
- İklim kanıt matrisi, fındık iklim tabloları, ERA5 ve ClimateGPT çıktıları
- V22 master, intervention ve actor tabloları
- Kaynak corpus ve animasyon için MongoDB kayıtları
- `backend/data/hhalys_hazelnut_event_climate_intervention_province_year_V22.csv`

Veri dosyaları, sanal ortamlar, node_modules ve bağlantı parolaları bu aktarımın dışında tutulmuştur. Kaynak türü, belge kimliği ve kanıt bağlantıları veriler hazırlanırken korunmalıdır.

## Yayınlama

Sunum sürümü için API'nin barındırılması veya onaylı verilerden tarayıcıda çalışan statik bir veri paketi hazırlanması gerekir. Mevcut uygulama bu adımlar tamamlanmadan GitHub Pages demosu olarak kabul edilmemelidir.
