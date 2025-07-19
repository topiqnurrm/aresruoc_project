import StoryIDB from '../../data/idb';
import ErrorHandler from '../../utils/error-handler';
import NetworkStatus from '../../utils/network-status';

class StoryCardPresenter {
  constructor(view) {
    this._view = view;
    this._storyIDB = StoryIDB;
    
    this._cleanupExpiredLocationCache();
  }

  /**
   * 
   * @param {string} storyId - ID cerita
   * @returns {Promise<boolean>} Status bookmark
   */
  async isStoryBookmarked(storyId) {
    try {
      return await this._storyIDB.isBookmarked(storyId);
    } catch (error) {
      console.error('Error checking bookmark status:', error);
      return false;
    }
  }

  /**
   * 
   * @param {Object} story - Objek cerita
   * @returns {Promise<boolean>} 
   */
  async addBookmark(story) {
    try {
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
   * @returns {Promise<boolean>} 
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
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @returns {Promise<string>} 
   */
  async getLocationName(lat, lon) {
    try {
      const initialCoords = this._formatCoordinates(lat, lon);
      this._view.updateLocationName(initialCoords);
      
      const cachedLocation = await this._storyIDB.getCachedLocation(lat, lon);
      if (cachedLocation) {
        if (cachedLocation.expiryTime && cachedLocation.expiryTime > Date.now()) {
          this._view.updateLocationName(cachedLocation.name);
          return cachedLocation.name;
        }
      }
      
      if (!NetworkStatus.isOnline) {
        return initialCoords;
      }
      
      try {
        const locationName = await Promise.race([
          this._fetchLocationName(lat, lon),
          new Promise((_, reject) => {
            setTimeout(() => {
              reject(new Error('Location fetch timeout'));
            }, 8000); 
          })
        ]);
        
        await this._storyIDB.cacheLocation(lat, lon, locationName);
        
        this._view.updateLocationName(locationName);
        
        return locationName;
      } catch (error) {
        console.warn('Location fetch failed:', error.message);
        
        await this._storyIDB.cacheLocation(lat, lon, initialCoords, 7);
        
        return initialCoords;
      }
    } catch (error) {
      console.error('Error getting location name:', error);
      return this._formatCoordinates(lat, lon);
    }
  }
  
  /**
   * 
   * @private
   */
  async _cleanupExpiredLocationCache() {
    try {
      setTimeout(async () => {
        const cleared = await this._storyIDB.clearExpiredLocationCache();
        if (cleared > 0) {
          console.log(`Cleared ${cleared} expired location entries from cache`);
        }
      }, 1000);
    } catch (error) {
      console.error('Error cleaning up location cache:', error);
    }
  }
  
  /**
   * 
   * @private
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @returns {Promise<string>} 
   */
  async _fetchLocationName(lat, lon) {
    try {
      if (!navigator.onLine) {
        throw new Error('Device is offline');
      }
      
      return await Promise.any([
        this._fetchFromNominatim(lat, lon),
        this._fetchFromBackupSource(lat, lon)
      ]);
    } catch (error) {
      console.warn('All geocoding sources failed:', error);
      return this._formatCoordinates(lat, lon);
    }
  }
  
  /**
   * 
   * @private
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @returns {Promise<string>} 
   */
  async _fetchFromNominatim(lat, lon) {
    try {
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&zoom=16`,
        {
          headers: {
            'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
            'User-Agent': 'StoryApp/1.0 (educational project)'
          },
          cache: 'force-cache'
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.address) {
        const parts = [];
        
        if (data.address.suburb) parts.push(data.address.suburb);
        else if (data.address.neighbourhood) parts.push(data.address.neighbourhood);
        else if (data.address.village) parts.push(data.address.village);
        
        if (data.address.city) parts.push(data.address.city);
        else if (data.address.town) parts.push(data.address.town);
        else if (data.address.county) parts.push(data.address.county);
        
        if (parts.length === 0) {
          return data.display_name.split(',').slice(0, 2).join(', ');
        } else {
          return parts.join(', ');
        }
      } else {
        throw new Error('No address data returned');
      }
    } catch (error) {
      console.warn('Nominatim geocoding failed:', error.message);
      throw error; 
    }
  }
  
  /**
   * 
   * @private
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @returns {Promise<string>} 
   */
  async _fetchFromBackupSource(lat, lon) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    throw new Error('Backup source not implemented');
  }
  
  /**
   *
   * @private
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @returns {string} 
   */
  _formatCoordinates(lat, lon) {
    return `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
  }
}

export default StoryCardPresenter;