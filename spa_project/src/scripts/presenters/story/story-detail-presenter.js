import StoryApi from '../../data/api';
import StoryIDB from '../../data/idb';
import ErrorHandler from '../../utils/error-handler';
import NetworkStatus from '../../utils/network-status';

export default class StoryDetailPresenter {
  constructor(view) {
    this._view = view;
    this._storyApi = StoryApi;
    this._storyIDB = StoryIDB;
  }

  async loadStoryDetail(id) {
    try {
      this._view.animateElements();
      
      if (!NetworkStatus.isOnline) {
        const cachedStory = await this._storyIDB.getStory(id);
        if (cachedStory) {
          const isBookmarked = await this._storyIDB.isBookmarked(id);
          this._view.updateStoryDetail(cachedStory, isBookmarked);
          return;
        } else {
          throw new Error('You are offline and this story is not available in your cache.');
        }
      }
      
      const response = await this._storyApi.getStoryDetail(id);
      
      if (response.story) {
        await this._storyIDB.putStory(response.story);
        
        const isBookmarked = await this._storyIDB.isBookmarked(id);
        
        this._view.updateStoryDetail(response.story, isBookmarked);
      } else {
        throw new Error('Failed to get story details');
      }
    } catch (error) {
      ErrorHandler.handleOfflineError(error, {
        offlineAction: async () => {
          const cachedStory = await this._storyIDB.getStory(id);
          if (cachedStory) {
            const isBookmarked = await this._storyIDB.isBookmarked(id);
            this._view.updateStoryDetail(cachedStory, isBookmarked);
          } else {
            this._view.showErrorState();
          }
        },
        onlineAction: () => {
          this._view.showErrorState();
        }
      });
    }
  }
  
  /**
   * 
   * @param {string} storyId - ID cerita
   * @returns {Promise<boolean>} Status operasi
   */
  async addBookmark(storyId) {
    try {
      const story = await this._storyIDB.getStory(storyId);
      if (!story) {
        throw new Error('Story not found');
      }
      
      await this._storyIDB.addBookmark(story);
      return true;
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Adding bookmark',
        defaultMessage: 'Failed to add bookmark'
      });
      return false;
    }
  }
  
  /**
   * 
   * @param {string} storyId - ID cerita
   * @returns {Promise<boolean>} Status operasi
   */
  async removeBookmark(storyId) {
    try {
      await this._storyIDB.removeBookmark(storyId);
      return true;
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Removing bookmark',
        defaultMessage: 'Failed to remove bookmark'
      });
      return false;
    }
  }
  
  /**
   * 
   * @param {string} storyId - ID cerita
   * @returns {Promise<boolean>} Status bookmark
   */
  async isBookmarked(storyId) {
    try {
      return await this._storyIDB.isBookmarked(storyId);
    } catch (error) {
      console.error('Error checking bookmark status:', error);
      return false;
    }
  }
}