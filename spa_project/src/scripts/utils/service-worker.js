import CONFIG from '../config';
import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { StaleWhileRevalidate, CacheFirst, NetworkFirst } from 'workbox-strategies';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';
import { ExpirationPlugin } from 'workbox-expiration';
import { BackgroundSyncPlugin } from 'workbox-background-sync';

const storyQueue = new BackgroundSyncPlugin('story-queue', {
  maxRetentionTime: 24 * 60, // Retry sampai 24 jam
  onSync: async ({ queue }) => {
    try {
      let entry;
      while ((entry = await queue.shiftRequest())) {
        try {
          const response = await fetch(entry.request.clone());
          console.log('Background sync successful for request:', entry.request.url);
          
          try {
            const responseData = await response.clone().json();
            if (responseData && !responseData.error && responseData.story) {
              const clients = await self.clients.matchAll({ type: 'window' });
              for (const client of clients) {
                client.postMessage({
                  type: 'STORY_SYNCED',
                  story: responseData.story
                });
              }
              
              self.registration.showNotification('Story Synced', {
                body: 'Your story has been successfully uploaded!',
                icon: '/images/icon-192.png',
                badge: '/images/icon-192.png' 
              });
            }
          } catch (parseError) {
            console.warn('Could not parse sync response:', parseError);
          }
        } catch (error) {
          console.error('Background sync failed for request:', error);
          await queue.unshiftRequest(entry);
          const retryCount = (entry.metadata?.retryCount || 0) + 1;
          entry.metadata = { ...entry.metadata, retryCount };
          
          if (retryCount > 5) {
            console.error('Too many retries, removing from queue:', entry);
            await queue.shiftRequest();
            
            const clients = await self.clients.matchAll({ type: 'window' });
            for (const client of clients) {
              client.postMessage({
                type: 'STORY_SYNC_FAILED'
              });
            }
          }
          break;
        }
      }
    } catch (error) {
      console.error('Fatal error in background sync:', error);
    }
  }
});

precacheAndRoute(self.__WB_MANIFEST);

registerRoute(
  ({ url, request }) => {
    return url.origin === CONFIG.BASE_URL && 
           url.pathname.includes('/stories') && 
           !url.pathname.includes('/addStory') &&
           request.destination !== 'image';
  },
  new NetworkFirst({
    cacheName: 'api-cache',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 24 * 60 * 60, // 1 day
      }),
    ],
    networkTimeoutSeconds: 3, 
  })
);


registerRoute(
  ({ request, url }) => 
    request.destination === 'image' ||
    (url.origin === CONFIG.BASE_URL && 
     (url.pathname.includes('/images') || url.pathname.includes('/stories/photoUrl'))),
  new StaleWhileRevalidate({
    cacheName: 'image-cache',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
      new ExpirationPlugin({
        maxEntries: 60,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
      }),
    ],
  })
);

registerRoute(
  ({ request, url }) => 
    url.origin === self.location.origin && 
    (request.destination === 'style' || 
     request.destination === 'script' || 
     request.destination === 'font'),
  new CacheFirst({
    cacheName: 'static-resources',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
      new ExpirationPlugin({
        maxEntries: 30,
        maxAgeSeconds: 60 * 60 * 24 * 30, // 30 days
      }),
    ],
  })
);

registerRoute(
  ({ url }) => url.origin === 'https://unpkg.com',
  new CacheFirst({
    cacheName: 'cdn-cache',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 20,
        maxAgeSeconds: 7 * 24 * 60 * 60, // 1 week
      }),
    ],
  })
);

self.addEventListener('sync', (event) => {
  if (event.tag === 'story-queue') {
    console.log('Sync event fired for story-queue');
  }
});

registerRoute(
  ({ url }) => 
    url.origin === CONFIG.BASE_URL && 
    (url.pathname.endsWith('/stories') || url.pathname.includes('/add-story')),
  async ({ event }) => {
    try {
      const response = await fetch(event.request.clone());
      return response;
    } catch (error) {
      return storyQueue.pushRequest({ request: event.request });
    }
  },
  'POST'
);

registerRoute(
  ({ request }) => request.mode === 'navigate',
  new NetworkFirst({
    cacheName: 'pages-cache',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
    ],
    networkTimeoutSeconds: 3,
  })
);

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  let payload;
  
  try {
    payload = event.data?.json();
  } catch (e) {
    const text = event.data ? event.data.text() : 'No content';
    payload = {
      title: 'New Notification',
      options: {
        body: text
      }
    };
  }
  
  payload = payload || CONFIG.DEFAULT_NOTIFICATION;
  
  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.options?.body || 'You have a new notification',
      icon: payload.options?.icon || '/images/icon-192.png',
      badge: payload.options?.badge || '/images/badge.png',
      data: { url: payload.options?.url || '/' }
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const urlToOpen = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then((clientList) => {
        if (clientList.length > 0) {
          return clientList[0].focus();
        }
        return clients.openWindow(urlToOpen);
      })
  );
});