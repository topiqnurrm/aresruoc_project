import { showAlert } from './index';

/**
 * Utility class for managing network status and providing consistent
 * online/offline state management across components
 */
class NetworkStatus {
  constructor() {
    this._isOnline = navigator.onLine;
    this._listeners = [];
    
    window.addEventListener('online', () => this._handleOnline());
    window.addEventListener('offline', () => this._handleOffline());
  }
  
  /**
   * Get current online status
   * @returns {boolean} Whether the app is online
   */
  get isOnline() {
    return navigator.onLine;
  }
  
  /**
   * Add a listener for network status changes
   * @param {Function} listener - Callback function(isOnline)
   * @returns {Function} Function to remove the listener
   */
  addListener(listener) {
    if (typeof listener !== 'function') return () => {};
    
    this._listeners.push(listener);
    
    return () => {
      this._listeners = this._listeners.filter(l => l !== listener);
    };
  }
  
  /**
   * Handle going online
   * @private
   */
  _handleOnline() {
    if (!this._isOnline) {
      this._isOnline = true;
      showAlert('You are back online', 'success');
      
      this._triggerBackgroundSync();
      
      this._notifyListeners();
    }
  }
  
  /**
   * Handle going offline
   * @private
   */
  _handleOffline() {
    if (this._isOnline) {
      this._isOnline = false;
      showAlert('You are offline. You can still browse cached content and your changes will sync when you\'re back online.', 'warning');
      
      this._notifyListeners();
    }
  }
  
  /**
   * Notify all listeners of the current network status
   * @private
   */
  _notifyListeners() {
    this._listeners.forEach(listener => {
      try {
        listener(this._isOnline);
      } catch (error) {
        console.error('Error in network status listener:', error);
      }
    });
  }
  
  /**
   * Trigger background sync when coming back online
   * @private
   */
  async _triggerBackgroundSync() {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.ready;
        if ('sync' in registration) {
          await registration.sync.register('story-queue');
        }
      } catch (error) {
        console.error('Error triggering background sync:', error);
      }
    }
  }
}

export default new NetworkStatus();