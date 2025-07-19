import { openDB } from 'idb';

const DB_NAME = 'StoryAppDB';
const DB_VERSION = 4; 
const STORE_NAME = 'stories';
const BOOKMARKS_STORE = 'bookmarks';
const OFFLINE_QUEUE_STORE = 'offlineQueue';
const LOCATION_CACHE_STORE = 'locationCache'; 

let dbInstance = null;

const dbPromise = async () => {
  if (dbInstance) {
    return dbInstance;
  }
  
  dbInstance = await openDB(DB_NAME, DB_VERSION, {
    upgrade(db, oldVersion, newVersion) {
      if (oldVersion < 1) {
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      }
      
      if (oldVersion < 2) {
        if (!db.objectStoreNames.contains(BOOKMARKS_STORE)) {
          db.createObjectStore(BOOKMARKS_STORE, { keyPath: 'id' });
        }
      }

      if (oldVersion < 3) {
        if (!db.objectStoreNames.contains(OFFLINE_QUEUE_STORE)) {
          const offlineStore = db.createObjectStore(OFFLINE_QUEUE_STORE, { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          offlineStore.createIndex('timestamp', 'timestamp');
          offlineStore.createIndex('status', 'status');
        }
      }
      
      if (oldVersion < 4) {
        if (!db.objectStoreNames.contains(LOCATION_CACHE_STORE)) {
          db.createObjectStore(LOCATION_CACHE_STORE, { keyPath: 'coordinates' });
        }
      }
    }
  });
  
  return dbInstance;
};

class StoryIDB {
  static async getAllStories() {
    try {
      const db = await dbPromise();
      return db.getAll(STORE_NAME);
    } catch (error) {
      console.error('Error getting all stories:', error);
      return [];
    }
  }

  static async getStory(id) {
    try {
      const db = await dbPromise();
      return db.get(STORE_NAME, id);
    } catch (error) {
      console.error('Error getting story:', error);
      return null;
    }
  }

  static async putStory(story) {
    try {
      const db = await dbPromise();
      return db.put(STORE_NAME, story);
    } catch (error) {
      console.error('Error putting story:', error);
      return false;
    }
  }

  static async deleteStory(id) {
    try {
      const db = await dbPromise();
      return db.delete(STORE_NAME, id);
    } catch (error) {
      console.error('Error deleting story:', error);
      return false;
    }
  }

  static async clearStories() {
    try {
      const db = await dbPromise();
      return db.clear(STORE_NAME);
    } catch (error) {
      console.error('Error clearing stories:', error);
      return false;
    }
  }

  static async syncWithAPI(apiStories) {
    try {
      const db = await dbPromise();
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      
      const existingStories = await store.getAll();
      
      const pendingStories = existingStories.filter(story => 
        story.id.toString().startsWith('offline-') || story.pending === true
      );
      
      for (const story of existingStories) {
        if (!story.id.toString().startsWith('offline-') && !story.pending) {
          await store.delete(story.id);
        }
      }
      
      for (const story of apiStories) {
        const matchingPendingStory = pendingStories.find(
          pending => pending.description === story.description && 
                    new Date(pending.createdAt).getTime() - new Date(story.createdAt).getTime() < 60000 
        );
        
        if (matchingPendingStory) {
          await store.delete(matchingPendingStory.id);
          
          await this._updateBookmarkReference(matchingPendingStory.id, story.id);
          
          console.log(`Replaced pending story (${matchingPendingStory.id}) with synced story (${story.id})`);
        }
        
        await store.put(story);
      }
      
      await tx.done;
      return true;
    } catch (error) {
      console.error('Error syncing with API:', error);
      return false;
    }
  }
  
  static async _updateBookmarkReference(oldId, newId) {
    try {
      const db = await dbPromise();
      const tx = db.transaction(BOOKMARKS_STORE, 'readwrite');
      const bookmark = await tx.store.get(oldId);
      
      if (bookmark) {
        const newStory = await this.getStory(newId);
        if (newStory) {
          await tx.store.delete(oldId);
          await tx.store.put({
            ...newStory,
            id: newId
          });
          console.log(`Updated bookmark reference from ${oldId} to ${newId}`);
        }
      }
      
      await tx.done;
    } catch (error) {
      console.error('Error updating bookmark reference:', error);
    }
  }

  static async getAllBookmarks() {
    try {
      const db = await dbPromise();
      return db.getAll(BOOKMARKS_STORE);
    } catch (error) {
      console.error('Error getting all bookmarks:', error);
      return [];
    }
  }

  static async getBookmark(id) {
    try {
      const db = await dbPromise();
      return db.get(BOOKMARKS_STORE, id);
    } catch (error) {
      console.error('Error getting bookmark:', error);
      return null;
    }
  }

  static async addBookmark(story) {
    try {
      const db = await dbPromise();
      return db.put(BOOKMARKS_STORE, story);
    } catch (error) {
      console.error('Error adding bookmark:', error);
      return false;
    }
  }

  static async removeBookmark(id) {
    try {
      const db = await dbPromise();
      return db.delete(BOOKMARKS_STORE, id);
    } catch (error) {
      console.error('Error removing bookmark:', error);
      return false;
    }
  }

  static async isBookmarked(id) {
    try {
      const bookmark = await this.getBookmark(id);
      return !!bookmark;
    } catch (error) {
      console.error('Error checking bookmark:', error);
      return false;
    }
  }

  static async saveForSync(operation) {
    try {
      const queueItem = {
        url: operation.url,
        method: operation.method,
        body: operation.body,
        timestamp: new Date().getTime(),
        status: 'pending'
      };
      
      const db = await dbPromise();
      return db.add(OFFLINE_QUEUE_STORE, queueItem);
    } catch (error) {
      console.error('Error saving for sync:', error);
      return false;
    }
  }
  
  static async getPendingOperations() {
    try {
      const db = await dbPromise();
      const tx = db.transaction(OFFLINE_QUEUE_STORE, 'readonly');
      const index = tx.store.index('status');
      return index.getAll('pending');
    } catch (error) {
      console.error('Error getting pending operations:', error);
      return [];
    }
  }
  
  static async markOperationComplete(id) {
    try {
      const db = await dbPromise();
      const tx = db.transaction(OFFLINE_QUEUE_STORE, 'readwrite');
      const item = await tx.store.get(id);
      
      if (item) {
        item.status = 'completed';
        await tx.store.put(item);
      }
      
      return tx.done;
    } catch (error) {
      console.error('Error marking operation complete:', error);
      return false;
    }
  }
  
  static async deleteOperation(id) {
    try {
      const db = await dbPromise();
      return db.delete(OFFLINE_QUEUE_STORE, id);
    } catch (error) {
      console.error('Error deleting operation:', error);
      return false;
    }
  }

  static async getCachedLocation(lat, lon) {
    try {
      const coordinateKey = this._formatCoordinateKey(lat, lon);
      const db = await dbPromise();
      return db.get(LOCATION_CACHE_STORE, coordinateKey);
    } catch (error) {
      console.error('Error getting cached location:', error);
      return null;
    }
  }
  
  static async cacheLocation(lat, lon, locationName, expiry = 30) {
    try {
      const coordinateKey = this._formatCoordinateKey(lat, lon);
      const location = {
        coordinates: coordinateKey,
        name: locationName,
        timestamp: Date.now(),
        expiryTime: Date.now() + (expiry * 24 * 60 * 60 * 1000)
      };
      
      const db = await dbPromise();
      await db.put(LOCATION_CACHE_STORE, location);
      return true;
    } catch (error) {
      console.error('Error caching location:', error);
      return false;
    }
  }
  
  static async getAllCachedLocations() {
    try {
      const db = await dbPromise();
      return db.getAll(LOCATION_CACHE_STORE);
    } catch (error) {
      console.error('Error getting all cached locations:', error);
      return [];
    }
  }
  
  static async clearExpiredLocationCache() {
    try {
      const db = await dbPromise();
      const tx = db.transaction(LOCATION_CACHE_STORE, 'readwrite');
      const store = tx.objectStore(LOCATION_CACHE_STORE);
      const locations = await store.getAll();
      
      const now = Date.now();
      let cleared = 0;
      
      for (const location of locations) {
        if (location.expiryTime && location.expiryTime < now) {
          await store.delete(location.coordinates);
          cleared++;
        }
      }
      
      await tx.done;
      
      if (cleared > 0) {
        console.log(`Cleared ${cleared} expired location cache entries`);
      }
      
      return cleared;
    } catch (error) {
      console.error('Error clearing expired location cache:', error);
      return 0;
    }
  }
  
  static _formatCoordinateKey(lat, lon) {
    if (typeof lat !== 'number' || typeof lon !== 'number') {
      throw new Error('Latitude and longitude must be numbers');
    }
    return `${lat.toFixed(6)},${lon.toFixed(6)}`;
  }
}

export default StoryIDB;