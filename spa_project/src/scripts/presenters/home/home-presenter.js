import StoryApi from '../../data/api';
import StoryIDB from '../../data/idb';
import { showAlert } from '../../utils';
import NetworkStatus from '../../utils/network-status';
import ErrorHandler from '../../utils/error-handler';

class HomePresenter {
  constructor(view) {
    this._view = view;
    this._storyApi = StoryApi;
    this._storyIDB = StoryIDB;
    
    this._networkStatusListener = NetworkStatus.addListener((isOnline) => {
      if (isOnline) {
        setTimeout(() => {
          this.fetchStories();
        }, 500);
      }
    });
    
    this._setupSyncListener();
  }

  _setupSyncListener() {
    navigator.serviceWorker.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'STORY_SYNCED') {
        showAlert('Your story has been uploaded successfully!', 'success');
        
        setTimeout(() => {
          this.fetchStories();
        }, 1000);
      }
    });
  }

  async fetchStories() {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        if (!this._view || !this._isViewMounted()) {
          console.warn('View is not mounted, skipping fetchStories');
          return;
        }
        
        this._view.showLoginRequired();
        return;
      }

      const cachedStories = await this._storyIDB.getAllStories();
      
      if (!this._view || !this._isViewMounted()) {
        console.warn('View is not mounted, skipping cached stories display');
        return;
      }
      
      if (cachedStories && cachedStories.length > 0) {
        this._view.displayStories(cachedStories);
      }

      if (NetworkStatus.isOnline) {
        try {
          const onlineStories = await this._storyApi.getAllStories();
          
          if (!this._view || !this._isViewMounted()) {
            console.warn('View is not mounted, skipping online stories display');
            return;
          }
          
          if (onlineStories && onlineStories.length > 0) {
            this._view.displayStories(onlineStories);
          } else {
            if (!cachedStories || cachedStories.length === 0) {
              this._view.showNoStories();
            }
          }
        } catch (apiError) {
          if (!this._view || !this._isViewMounted()) {
            console.warn('View is not mounted, skipping error handling');
            return;
          }
          
          ErrorHandler.handleApiError(apiError, {
            context: 'Fetching stories',
            defaultMessage: 'Failed to load stories. Showing cached content.'
          });
          
          if (!cachedStories || cachedStories.length === 0) {
            const fallbackStories = await this._storyIDB.getAllStories();
            
            if (!this._view || !this._isViewMounted()) {
              return;
            }
            
            if (fallbackStories.length > 0) {
              this._view.displayStories(fallbackStories);
            } else {
              throw apiError;
            }
          }
        }
      } else {
        if (!this._view || !this._isViewMounted()) {
          return;
        }
        
        if (cachedStories && cachedStories.length > 0) {
          showAlert('You are offline. Showing cached stories.', 'info');
        } else {
          this._view.showError('You are offline and no cached stories are available.');
        }
      }
    } catch (error) {
      if (!this._view || !this._isViewMounted()) {
        return;
      }
      
      ErrorHandler.handleApiError(error, {
        context: 'Story loading',
        defaultMessage: 'Failed to load stories. Please try again later.'
      });
      this._view.showError(error.message || 'Failed to load stories. Please try again later.');
    }
  }

  async fetchStoriesWithLocation() {
    try {
      if (NetworkStatus.isOnline) {
        try {
          const storiesWithLocation = await this._storyApi.getAllStories(true);
          return storiesWithLocation;
        } catch (apiError) {
          ErrorHandler.handleApiError(apiError, {
            context: 'Fetching map locations',
            showNotification: false
          });
          throw apiError;
        }
      } else {
        const allCachedStories = await this._storyIDB.getAllStories();
        const cachedStoriesWithLocation = allCachedStories.filter(
          story => story.lat && story.lon
        );
        
        if (cachedStoriesWithLocation.length > 0) {
          showAlert('Showing cached story locations', 'info');
          return cachedStoriesWithLocation;
        } else {
          throw new Error('No cached story locations available');
        }
      }
    } catch (error) {
      if (this._view && this._isViewMounted()) {
        ErrorHandler.handleOfflineError(error, {
          offlineAction: () => this._view.showMapError('You are offline. No cached map data available.')
        });
      }
      return [];
    }
  }
  
  /**
   * 
   * @private
   * @returns {boolean} 
   */
  _isViewMounted() {
    if (this._view._storyListElement) {
      return document.body.contains(this._view._storyListElement);
    }
    return false;
  }
  
  cleanup() {
    if (this._networkStatusListener) {
      this._networkStatusListener();
    }
  }
}

export default HomePresenter;