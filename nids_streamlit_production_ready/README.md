# NIDS Streamlit Studio — Production-like Package

Paket ini memisahkan mode deploy **Cloud** dan **Local Full**.

## Struktur utama
- `app.py` → entrypoint **Streamlit Cloud** (ML-first, lebih ringan)
- `app_full_local.py` → entrypoint **lokal penuh** (fitur lebih lengkap)
- `dashboard_cloud.py` → dashboard cloud-friendly
- `pipeline_cloud.py` → pipeline cloud-friendly
- `dashboard_local.py` → dashboard lokal penuh
- `pipeline_local.py` → pipeline lokal penuh
- `requirements.txt` → dependency default untuk **Cloud**
- `requirements-cloud.txt` → salinan dependency Cloud
- `requirements-local.txt` → dependency lokal penuh (termasuk TensorFlow)

## Jalankan lokal (mode penuh)
```bash
pip install -r requirements-local.txt
streamlit run app_full_local.py
```

## Jalankan lokal (mode cloud-simulation)
```bash
pip install -r requirements-cloud.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
Gunakan file berikut saat membuat app:
- **Entrypoint**: `app.py`
- **Requirements**: `requirements.txt`

Mode Cloud sengaja dibuat **ML-first** dan lebih aman untuk environment Streamlit Community Cloud.
Untuk eksperimen DL berat, gunakan `app_full_local.py` di mesin lokal.

## Catatan
- Semua path diset relatif terhadap root repo.
- Simpan dataset dengan path relatif di dalam repo, atau upload melalui dashboard.
- Untuk Cloud, hindari dependency macOS-specific seperti `tensorflow-metal`.
