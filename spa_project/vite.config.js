import { defineConfig } from 'vite';
import { resolve } from 'path';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  root: resolve(__dirname, 'src'),
  publicDir: resolve(__dirname, 'src', 'public'),
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  base:"/spa_project/",
  plugins: [
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['favicon.png', 'images/*.png'],
        manifest: {
          name: 'SPA Project',
          short_name: 'SPAProject',
          description: "share lock i'll come there",
          theme_color: '#4a90e2',
          icons: [
            {
              src: '/images/icon-192.png',
              sizes: '192x192',
              type: 'image/png'
            },
            {
              src: '/images/icon-512.png',
              sizes: '512x512',
              type: 'image/png'
            }
          ]
        },
        strategies: 'injectManifest',
        srcDir: 'scripts/utils',
        filename: 'service-worker.js',
        injectManifest: {
          globPatterns: ['**/*.{js,css,html,png,jpg,jpeg,svg,json}']
        },
        workbox: {
          runtimeCaching: [
            {
              urlPattern: ({ url }) => url.origin === 'https://story-api.dicoding.dev',
              handler: 'NetworkFirst',
              options: {
                cacheName: 'api-cache',
                expiration: {
                  maxEntries: 50,
                  maxAgeSeconds: 24 * 60 * 60 // 1 day
                },
                networkTimeoutSeconds: 3
              }
            },
            {
              urlPattern: ({ request }) => request.destination === 'image',
              handler: 'StaleWhileRevalidate',
              options: {
                cacheName: 'image-cache',
                expiration: {
                  maxEntries: 60,
                  maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
                }
              }
            }
          ]
        },
        devOptions: {
          enabled: true,
          type: 'module'
        }
      })
    ]
});
