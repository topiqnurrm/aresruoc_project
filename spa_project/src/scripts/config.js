const CONFIG = {
  BASE_URL: 'https://story-api.dicoding.dev/v1',
  
  ENDPOINTS: {
    REGISTER: '/register',
    LOGIN: '/login',
    STORIES: '/stories',
    DETAIL_STORY: (id) => `/stories/${id}`,
    ADD_STORY: '/stories',
    
    NOTIFICATIONS: {
      SUBSCRIBE: '/notifications/subscribe',
      SUBSCRIBE_USER: (userId) => `/notifications/subscribe/${userId}`,
      UNSUBSCRIBE: '/notifications/subscribe', 
    },
  },
  
  PUSH_MSG_VAPID_PUBLIC_KEY: 'BCCs2eonMI-6H2ctvFaWg-UYdDv387Vno_bzUzALpB442r2lCnsHmtrx8biyPi_E-1fSGABK_Qs_GlvPoJJqxbk',
  
  DEFAULT_NOTIFICATION: {
    title: 'New Notification',
    options: {
      body: 'You have a new notification',
      icon: '/images/icon-192.png',
      badge: '/images/badge.png'
    }
  },
  
  CACHE_NAME: 'StoryApp-V1',
};

export default CONFIG;
