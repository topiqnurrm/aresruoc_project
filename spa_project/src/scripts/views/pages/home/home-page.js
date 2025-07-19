import HomePresenter from '../../../presenters/home/home-presenter';
import { createSkeletonStory, initMapWithLayerControl } from '../../../utils';
import '../story/story-list';
import AnimationHelper from '../../../utils/animation-helper';
import ErrorHandler from '../../../utils/error-handler';

export default class HomePage {
  constructor() {
    this._stories = [];
    this._map = null;
    this._storyListElement = null;
    this._presenter = new HomePresenter(this);
  }

  async render() {
    return `
      <section class="hero">
        <div class="hero__inner">
          <h1 class="hero__title">Share Your Story</h1>
          <p class="hero__tagline">Tell your amazing experiences and inspire others</p>
          <a href="#/add-story" class="hero__button">Add Your Story</a>
        </div>
      </section>
      
      <div class="container">
        <section class="content-section map-section">
          <h2 class="section-title">Story Locations</h2>
          <div id="map" style="height: 400px;"></div>
        </section>
        
        <section class="content-section stories-section">
          <h2 class="section-title">Recent Stories</h2>
          <div id="story-list-container">
            <story-list class="story-list">
              ${createSkeletonStory()}
            </story-list>
          </div>
        </section>
      </div>
    `;
  }

  async afterRender() {
    const storyListElement = document.querySelector('story-list');
    this._storyListElement = storyListElement;
    
    try {
      await this._presenter.fetchStories();
      
      this._initializeMap();
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Home page initialization',
        defaultMessage: 'Failed to initialize home page'
      });
    }
    
    window.addEventListener('hashchange', this._cleanup.bind(this));
  }
  
  _cleanup() {
    if (this._presenter) {
      this._presenter.cleanup();
    }
    
    if (this._map) {
      this._map.remove();
      this._map = null;
    }
    
    window.removeEventListener('hashchange', this._cleanup.bind(this));
  }

  displayStories(stories) {
    this._stories = stories;
    
    if (!this._storyListElement || !document.body.contains(this._storyListElement)) {
      const container = document.getElementById('story-list-container');
      if (container) {
        container.innerHTML = '';
        
        this._storyListElement = document.createElement('story-list');
        this._storyListElement.className = 'story-list';
        container.appendChild(this._storyListElement);
      } else {
        console.warn('Story list container not found');
        return;
      }
    }
    
    this._storyListElement.stories = this._stories;
    
    setTimeout(() => {
      const storyItems = document.querySelectorAll('.story-item');
      storyItems.forEach((item, index) => {
        AnimationHelper.animateElement(item, {
          delay: 100 + (index * 50),
          transform: 'translateY(0)',
          opacity: '1'
        });
      });
    }, 100);
  }

  showNoStories() {
    if (!this._storyListElement || !document.body.contains(this._storyListElement)) {
      const container = document.getElementById('story-list-container');
      if (container) {
        container.innerHTML = `
          <div class="no-stories">
            <p>No stories available. Be the first to share!</p>
          </div>
        `;
      }
      return;
    }
    
    this._storyListElement.innerHTML = `
      <div class="no-stories">
        <p>No stories available. Be the first to share!</p>
      </div>
    `;
  }

  showLoginRequired() {
    if (!this._storyListElement || !document.body.contains(this._storyListElement)) {
      const container = document.getElementById('story-list-container');
      if (container) {
        container.innerHTML = `
          <div class="error-message">
            <p>You need to login first to view stories.</p>
            <a href="#/login" class="login-link">Click here to login</a>
          </div>
        `;
      }
      return;
    }
    
    this._storyListElement.innerHTML = `
      <div class="error-message">
        <p>You need to login first to view stories.</p>
        <a href="#/login" class="login-link">Click here to login</a>
      </div>
    `;
  }

  showError(message) {
    if (!this._storyListElement || !document.body.contains(this._storyListElement)) {
      const container = document.getElementById('story-list-container');
      if (container) {
        container.innerHTML = `
          <div class="error-message">
            <p>${message}</p>
            <button class="retry-button" id="retry-button">Retry</button>
          </div>
        `;
        
        const retryButton = document.getElementById('retry-button');
        if (retryButton) {
          retryButton.addEventListener('click', () => {
            container.innerHTML = createSkeletonStory();
            this._presenter.fetchStories();
          });
        }
      }
      return;
    }
    
    this._storyListElement.innerHTML = `
      <div class="error-message">
        <p>${message}</p>
        <button class="retry-button" id="retry-button">Retry</button>
      </div>
    `;
    
    const retryButton = document.getElementById('retry-button');
    if (retryButton) {
      retryButton.addEventListener('click', () => {
        this._storyListElement.innerHTML = createSkeletonStory();
        this._presenter.fetchStories();
      });
    }
  }

  showMapError(message) {
    const mapElement = document.getElementById('map');
    if (mapElement) {
      mapElement.innerHTML = `
        <div class="map-error">
          <p>${message}</p>
        </div>
      `;
      mapElement.classList.add('map-error-container');
    }
  }

  async _initializeMap() {
    try {
      const defaultLat = -6.2088;
      const defaultLng = 106.8456;
      
      const mapElement = document.getElementById('map');
      if (!mapElement) {
        console.warn('Map element not found');
        return;
      }
      
      this._map = initMapWithLayerControl('map', defaultLat, defaultLng);
      
      const storiesWithLocation = await this._presenter.fetchStoriesWithLocation();
      
      this._displayStoriesOnMap(storiesWithLocation);
      
      setTimeout(() => {
        if (this._map) {
          this._map.invalidateSize();
        }
      }, 100);
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Map initialization',
        defaultMessage: 'Failed to load the map',
        showNotification: false
      });
      this.showMapError('Failed to load the map. Please try again later.');
    }
  }
  
  _displayStoriesOnMap(storiesWithLocation) {
    if (!this._map) return;
    
    const markers = [];
    storiesWithLocation.forEach((story) => {
      if (story.lat && story.lon) {
        const marker = L.marker([story.lat, story.lon]).addTo(this._map);
        
        const popupContent = `
          <div class="story-popup">
            <h3>${story.name}</h3>
            <img src="${story.photoUrl}" alt="story by ${story.name}" style="max-width:100px; max-height:100px; margin:5px 0;">
            <p>${story.description.substring(0, 100)}${story.description.length > 100 ? '...' : ''}</p>
            <p class="story-date">Posted on: ${new Date(story.createdAt).toLocaleDateString()}</p>
            <a href="#/stories/${story.id}" class="story-link">View full story</a>
          </div>
        `;
        
        marker.bindPopup(popupContent);
        markers.push(marker);
      }
    });
    
    if (markers.length > 0 && this._map) {
      const group = L.featureGroup(markers);
      this._map.fitBounds(group.getBounds().pad(0.1));
    }
  }
}