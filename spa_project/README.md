# SPA Project

Progressive Web App (PWA) ini memungkinkan user berbagi pengalaman mereka melalui cerita, menambahkan foto, dan menandai lokasi cerita pada peta. Aplikasi ini dibangun hasil dari submission kelas Belajar Pengembangan Web Intermediate tingkat mahir dalam learning path Front-End Web Developer.


## Architecture

Proyek ini mengikuti Model-View-Presenter pattern (MVP):

- **Model (Data)**: Ditangani oleh `data/api.js` dan `data/idb.js`
- **View**: Dikelola oleh komponen dalam direktori `views`
- **Presenter**: Business logic dalam direktori `presenters`

## Fitur

### 1. User Authentication

- **Register**: Pengguna baru dapat membuat akun dengan email dan kata sandi
- **Login**: Pengguna yang ada dapat mengakses akun mereka
- **Authentication State Management**: Status login persisten menggunakan token localStorage

### 2. Story Management

- **Lihat Cerita**: Menjelajahi cerita dari pengguna lain
- **Buat Cerita**: Tambahkan cerita baru dengan deskripsi dan foto
- **Unggah Foto**: Unggah gambar dari kamera atau sistem file max 1MB
- **Penandaan Lokasi**: Tambahkan koordinat geografis ke cerita

### 3. Offline Capabilities

- **Akses Offline**: Lihat cerita yang telah dimuat sebelumnya saat offline
- **Background Sync**: Buat cerita offline dan sinkronkan saat kembali online
- **Data Persistence**: Penyimpanan IndexedDB untuk data offline

### 4. Bookmark

- **Simpan Bookmark**: Simpan cerita favorit untuk dilihat nanti
- **Kelola Bookmark**: Lihat dan hapus cerita yang telah di-bookmark

### 5. Lokasi

- **Tampilan Peta**: Lihat lokasi cerita pada peta interaktif
- **Geocoding**: Secara otomatis mengambil nama lokasi dari koordinat
- **Pemilihan Lokasi**: Pilih lokasi cerita dengan mengklik pada peta atau menggunakan lokasi perangkat

### 6. Progressive Web App

- **Dapat Diinstal**: Dapat diinstal pada perangkat seperti aplikasi native
- **Offline-first**: Berfungsi tanpa koneksi internet
- **Notifikasi Push**: Dukungan notifikasi untuk cerita baru dan pembaruan
- **Desain Responsif**: Beradaptasi dengan berbagai ukuran layar

## Teknologies dan Tools

### Library

- **Leaflet.js**: Peta interaktif untuk fitur lokasi
- **SweetAlert2**: Alert dan Notification
- **Font Awesome**: Toolkit ikon

### Teknologi PWA

- **Service Workers**: Untuk kemampuan offline dan caching
- **IndexedDB (via idb)**: Penyimpanan sisi klien untuk data offline
- **Web App Manifest**: Untuk fungsionalitas PWA yang dapat diinstal

### Alat Build

- **Vite**
- **Workbox**: Generasi service worker dan optimasi PWA
- **vite-plugin-pwa**: Integrasi PWA untuk Vite

### Integrasi API

- **Fetch API**: Untuk permintaan jaringan
- **Background Sync API**: Untuk sinkronisasi data offline ke online
- **Geolocation API**: Untuk deteksi lokasi perangkat
- **Geocoding API**: Untuk mengubah koordinat menjadi nama lokasi


## Build untuk Production

1. Build proyek:

```bash
npm run build
```

2. Output build akan berada di direktori `dist`

3. Pratinjau / preview build produksi secara lokal:

```bash
npm run preview
```

## Struktur Proyek

```
/story
│
├── package.json          # Dependensi dan scripts
├── vite.config.js        # Vite configuration
├── workbox-config.js     # Service worker configuration
│
├── src/
│   ├── index.html        # Main HTML entry point
│   │
│   ├── public/           # Static files
│   │   ├── favicon.png
│   │   ├── manifest.json # Manifest PWA
│   │   ├── images/       # Gambar dan ikon aplikasi
│   │   └── screenshots/  # Screenshot aplikasi
│   │
│   ├── scripts/
│   │   ├── config.js     # App configuration
│   │   ├── index.js      # JavaScript entry point
│   │   │
│   │   ├── data/         # Data management / Model
│   │   │   ├── api.js    # Interaksi API
│   │   │   └── idb.js    # IndexedDB operations
│   │   │
│   │   ├── presenters/   # Business logic
│   │   │   ├── auth/     # Authentication presenters
│   │   │   ├── bookmarks/# Bookmark management
│   │   │   ├── guest/    # Guest features
│   │   │   ├── home/     # Home page logic
│   │   │   └── story/    # Story management
│   │   │
│   │   ├── routes/       # Client-side routing
│   │   │   ├── routes.js
│   │   │   └── url-parser.js
│   │   │
│   │   └── utils/        # Utility
│   │       ├── animation-helper.js
│   │       ├── camera-helper.js
│   │       ├── error-handler.js
│   │       ├── map-location-helper.js
│   │       ├── network-status.js
│   │       ├── notification-helper.js
│   │       ├── service-worker.js
│   │       └── story-card-helper.js
│   │
│   ├── views/            # UI Pages
│   │   ├── app.js        # Main app container
│   │   │
│   │   └── pages/        # App pages
│   │       ├── auth/     # Login and register
│   │       ├── bookmarks/# Bookmarks page
│   │       ├── guest/    # Guest page
│   │       ├── home/     # Home page
│   │       └── story/    # Story pages
│   │
│   └── styles/           # CSS
│       ├── main.css      # Main stylesheet
│       ├── base/         # Base styles
│       ├── components/   # Component styles
│       ├── layout/       # Layout styles
│       ├── pages/        # Page-specific styles
│       └── utils/        # Utility styles
```
