import CONFIG from '../config';
import { showAlert } from '../utils';
import StoryIDB from './idb';
import NetworkStatus from '../utils/network-status';
import ErrorHandler from '../utils/error-handler';

class StoryApi {
  static async _fetchWithAuth(endpoint, options = {}) {
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    const headers = {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    };

    const response = await fetch(`${CONFIG.BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  static async _fetchWithoutAuth(endpoint, options = {}) {
    const response = await fetch(`${CONFIG.BASE_URL}${endpoint}`, options);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  static async getAllStories(withLocation = false) {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found');
      }
  
      const locationParam = withLocation ? '?location=1' : '';
      const url = `${CONFIG.BASE_URL}/stories${locationParam}`;
      
      if (!NetworkStatus.isOnline) {
        const offlineStories = await StoryIDB.getAllStories();
        if (offlineStories.length > 0) {
          showAlert('You are offline. Showing cached stories.', 'info');
          return offlineStories;
        }
        throw new Error('You are offline and no cached stories are available');
      }
      
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const responseJson = await response.json();
      
      if (!responseJson.error && responseJson.listStory) {
        await StoryIDB.syncWithAPI(responseJson.listStory);
        return responseJson.listStory;
      } else {
        throw new Error(responseJson.message || 'Failed to fetch stories');
      }
    } catch (error) {
      return ErrorHandler.handleOfflineError(error, {
        offlineAction: async () => {
          const offlineStories = await StoryIDB.getAllStories();
          if (offlineStories.length > 0) {
            return offlineStories;
          }
          throw error;
        }
      });
    }
  }

  static async getStoryDetail(id) {
    try {
      const token = localStorage.getItem('token');
      const url = `${CONFIG.BASE_URL}/stories/${id}`;
      
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const responseJson = await response.json();
      
      if (!responseJson.error && responseJson.story) {
        return responseJson; 
      } else {
        throw new Error(responseJson.message || 'Failed to fetch story detail');
      }
    } catch (error) {
      console.error('Failed to fetch story detail:', error);
      
      if (!navigator.onLine || error.message.includes('Failed to fetch')) {
        const cachedStory = await StoryIDB.getStory(id);
        if (cachedStory) {
          showAlert('Showing cached story content', 'info');
          return { story: cachedStory };
        }
      }
      
      showAlert('Failed to load story detail. Please try again.', 'error');
      throw error; 
    }
  }

  static async addStory({ description, photo, lat, lon }) {
    try {
      const formData = new FormData();
      formData.append('description', description);
      formData.append('photo', photo);
      if (lat) formData.append('lat', lat);
      if (lon) formData.append('lon', lon);

      const token = localStorage.getItem('token');
      const url = `${CONFIG.BASE_URL}/stories`;
      
      if (!navigator.onLine) {
        const photoData = await this._convertBlobToBase64(photo);
        await StoryIDB.saveForSync({
          type: 'add-story',
          url: url,
          method: 'POST',
          body: {
            description,
            photo: photoData,
            lat: lat || null,
            lon: lon || null,
            token
          }
        });
        
        const tempStory = {
          id: `offline-${Date.now()}`,
          description,
          photoUrl: URL.createObjectURL(photo),
          createdAt: new Date().toISOString(),
          lat: lat || null,
          lon: lon || null,
          name: localStorage.getItem('name') || 'User',
          pending: true
        };
        
        await StoryIDB.putStory(tempStory);
        showAlert('Story saved offline and will be uploaded when you are back online', 'info');
        return { error: false, message: 'Story saved for sync', story: tempStory };
      }
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseJson = await response.json();
      
      if (!responseJson.error) {
        showAlert('Story added successfully!');
        return responseJson;
      } else {
        throw new Error(responseJson.message || 'Failed to add story');
      }
    } catch (error) {
      console.error('Failed to add story:', error);
      showAlert('Failed to add story. Please try again.', 'error');
      throw error;
    }
  }

  static async _convertBlobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  static async register({ name, email, password }) {
    try {
      const response = await fetch(`${CONFIG.BASE_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, email, password }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseJson = await response.json();
      
      if (!responseJson.error) {
        showAlert('Registration successful! Please login.');
        return responseJson;
      } else {
        throw new Error(responseJson.message || 'Registration failed');
      }
    } catch (error) {
      console.error('Registration failed:', error);
      showAlert('Registration failed. Please try again.', 'error');
      throw error;
    }
  }

  static async login({ email, password }) {
    try {
      const response = await fetch(`${CONFIG.BASE_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseJson = await response.json();
      
      if (!responseJson.error && responseJson.loginResult) {
        localStorage.setItem('token', responseJson.loginResult.token);
        localStorage.setItem('name', responseJson.loginResult.name); 
        document.dispatchEvent(new Event('authChanged'));
        showAlert('Login successful!');
        return responseJson;
      } else {
        throw new Error(responseJson.message || 'Login failed');
      }
    } catch (error) {
      console.error('Login failed:', error);
      showAlert('Login failed. Please check your credentials.', 'error');
      throw error;
    }
  }

  static async addGuestStory({ description, photo, lat, lon }) {
    try {
      const formData = new FormData();
      formData.append('description', description);
      formData.append('photo', photo);
      if (lat) formData.append('lat', lat);
      if (lon) formData.append('lon', lon);

      const url = `${CONFIG.BASE_URL}/stories/guest`;

      if (!navigator.onLine) {
        const photoData = await this._convertBlobToBase64(photo);
        await StoryIDB.saveForSync({
          type: 'add-guest-story',
          url: url,
          method: 'POST',
          body: {
            description,
            photo: photoData,
            lat: lat || null,
            lon: lon || null
          }
        });
        
        const tempStory = {
          id: `offline-guest-${Date.now()}`,
          description,
          photoUrl: URL.createObjectURL(photo),
          createdAt: new Date().toISOString(),
          lat: lat || null,
          lon: lon || null,
          name: 'Guest (pending)',
          pending: true
        };
        
        await StoryIDB.putStory(tempStory);
        showAlert('Guest story saved offline and will be uploaded when you are back online', 'info');
        return { error: false, message: 'Story saved for sync', story: tempStory };
      }
      
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseJson = await response.json();
      
      if (!responseJson.error) {
        return responseJson;
      } else {
        throw new Error(responseJson.message || 'Failed to add guest story');
      }
    } catch (error) {
      console.error('Failed to add guest story:', error);
      showAlert('Failed to add story. Please try again.', 'error');
      throw error;
    }
  }

  static async getGuestStories() {
    try {
      const url = `${CONFIG.BASE_URL}/stories/guest`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseJson = await response.json();
      
      if (!responseJson.error && responseJson.listStory) {
        const stories = responseJson.listStory.map(story => ({
          ...story,
          name: "Guest" 
        }));
        
        return stories;
      } else {
        throw new Error(responseJson.message || 'Failed to fetch guest stories');
      }
    } catch (error) {
      console.error('Failed to fetch guest stories:', error);
      
      if (!navigator.onLine || error.message.includes('Failed to fetch')) {
        const allStories = await StoryIDB.getAllStories();
        const guestStories = allStories.filter(story => 
          story.name === 'Guest' || story.name === 'Guest (pending)'
        );
        
        if (guestStories.length > 0) {
          showAlert('Showing cached guest stories', 'info');
          return guestStories;
        }
      }
      
      showAlert('Failed to load guest stories. Please try again.', 'error');
      return []; 
    }
  }
}

export default StoryApi;