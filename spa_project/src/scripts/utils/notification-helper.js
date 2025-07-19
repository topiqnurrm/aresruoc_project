import CONFIG from '../config';

const VAPID_PUBLIC_KEY = CONFIG.PUSH_MSG_VAPID_PUBLIC_KEY;

class NotificationHelper {
  static async checkPermission() {
    if (!('Notification' in window)) {
      console.log('Browser does not support notifications');
      return false;
    }
    
    if (Notification.permission === 'granted') {
      return true;
    }
    
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }
  
  static async registerServiceWorker() {
    if (!('serviceWorker' in navigator)) {
      console.log('Service Worker not supported');
      return false;
    }
    
    try {
      const registration = await navigator.serviceWorker.register('/service-worker.js');
      console.log('Service Worker registered');
      return registration;
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      return false;
    }
  }
  
  static async subscribeToPush(registration) {
    try {
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: this.urlBase64ToUint8Array(VAPID_PUBLIC_KEY)
      });
      
      return subscription;
    } catch (error) {
      console.error('Failed to subscribe to push:', error);
      return null;
    }
  }
  
  static async sendSubscriptionToServer(subscription) {
    const token = localStorage.getItem('token');
    if (!token) return false;
    
    try {
      const response = await fetch(`${CONFIG.BASE_URL}${CONFIG.ENDPOINTS.NOTIFICATIONS.SUBSCRIBE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          endpoint: subscription.endpoint,
          keys: {
            p256dh: subscription.toJSON().keys.p256dh,
            auth: subscription.toJSON().keys.auth
          }
        })
      });
      
      return response.ok;
    } catch (error) {
      console.error('Error sending subscription to server:', error);
      return false;
    }
  }
  
  static urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, '+')
      .replace(/_/g, '/');
    
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    
    return outputArray;
  }
  
  static async showNotification(title, options) {
    if (!await this.checkPermission()) return;
    
    const registration = await navigator.serviceWorker.ready;
    registration.showNotification(title, options);
  }
  
  static async initPushNotifications() {
    if (!await this.checkPermission()) return false;
    
    const registration = await this.registerServiceWorker();
    if (!registration) return false;
    
    const subscription = await this.subscribeToPush(registration);
    if (!subscription) return false;
    
    return await this.sendSubscriptionToServer(subscription);
  }
  
  static async notifyStoryCreated(description, isGuest = false) {
    if (!await this.checkPermission()) return false;
    
    try {
      const shortenedDescription = description.length > 30 
        ? `${description.substring(0, 30)}...` 
        : description;
      
      const notificationData = {
        title: isGuest ? "Guest story berhasil dibuat" : "Story berhasil dibuat",
        options: {
          body: isGuest 
            ? `Story tamu telah dibuat dengan deskripsi: ${shortenedDescription}` 
            : `Anda telah membuat story baru dengan deskripsi: ${shortenedDescription}`
        }
      };
      
      await this.showNotification(notificationData.title, notificationData.options);
      
      return true;
    } catch (error) {
      console.error('Failed to show story creation notification:', error);
      return false;
    }
  }
  
  static async unsubscribeFromPush() {
    if (!('serviceWorker' in navigator)) return false;
    
    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();
      
      if (!subscription) return true;
      
      await this.sendUnsubscribeToServer(subscription);
      await subscription.unsubscribe();
      
      return true;
    } catch (error) {
      console.error('Failed to unsubscribe from push:', error);
      return false;
    }
  }
  
  static async sendUnsubscribeToServer(subscription) {
    const token = localStorage.getItem('token');
    if (!token) return false;
    
    try {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('DELETE', `${CONFIG.BASE_URL}${CONFIG.ENDPOINTS.NOTIFICATIONS.UNSUBSCRIBE}`, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        
        xhr.onload = function() {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(true);
          } else {
            reject(new Error(`HTTP error! status: ${xhr.status}`));
          }
        };
        
        xhr.onerror = function() {
          reject(new Error('Network error occurred'));
        };
        
        xhr.send(JSON.stringify({
          endpoint: subscription.endpoint
        }));
      });
    } catch (error) {
      console.error('Error sending unsubscription to server:', error);
      return false;
    }
  }
  
  static async isSubscribed() {
    if (!('serviceWorker' in navigator)) return false;
    
    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();
      
      return !!subscription;
    } catch (error) {
      console.error('Error checking subscription status:', error);
      return false;
    }
  }
  
  static async togglePushNotification() {
    const isSubscribed = await this.isSubscribed();
    
    if (isSubscribed) {
      return await this.unsubscribeFromPush();
    } else {
      return await this.initPushNotifications();
    }
  }
  
  static async toggleNavPushNotification() {
    const isSubscribed = await this.isSubscribed();
    
    if (isSubscribed) {
      return await this.unsubscribeFromPush();
    } else {
      return await this.initPushNotifications();
    }
  }
}

export default NotificationHelper;