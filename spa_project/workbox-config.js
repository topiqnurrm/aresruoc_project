module.exports = {
    globDirectory: 'dist/',
    globPatterns: [
      '**/*.{html,js,css,png,jpg,jpeg,svg,json}'
    ],
    swDest: 'dist/service-worker.js',
    runtimeCaching: [
      {
        urlPattern: /^https:\/\/story-api\.dicoding\.dev\/v1\/(?!.*images).*/,
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
        urlPattern: /^https:\/\/story-api\.dicoding\.dev\/v1\/.*(?:images|\/stories\/photoUrl).*/,
        handler: 'StaleWhileRevalidate',
        options: {
          cacheName: 'image-cache',
          expiration: {
            maxEntries: 60,
            maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
          }
        }
      },
      {
        urlPattern: /^https:\/\/unpkg\.com\/.*/,
        handler: 'CacheFirst',
        options: {
          cacheName: 'cdn-cache',
          expiration: {
            maxEntries: 20,
            maxAgeSeconds: 7 * 24 * 60 * 60 // 1 week
          }
        }
      },
      {
        urlPattern: /\.(?:js|css|html)$/,
        handler: 'CacheFirst',
        options: {
          cacheName: 'static-resources',
          expiration: {
            maxEntries: 30,
            maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
          }
        }
      }
    ]
  };