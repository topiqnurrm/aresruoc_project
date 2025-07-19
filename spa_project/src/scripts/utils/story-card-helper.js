import { showAlert, showFormattedDate } from './index';
import AnimationHelper from './animation-helper';
import StoryCardPresenter from '../presenters/story/story-card-presenter';

class StoryCardHelper extends HTMLElement {
  constructor() {
    super();
    this._story = null;
    this._locationName = 'Unknown location';
    this._isBookmarked = false;
    this._presenter = new StoryCardPresenter(this);
  }
  
  set story(story) {
    this._story = story;
    this._checkBookmarkStatus();
  }
  
  async _checkBookmarkStatus() {
    if (this._story) {
      this._isBookmarked = await this._presenter.isStoryBookmarked(this._story.id);
      this.render();
      this._fetchLocationName();
    }
  }
  
  render() {
    if (!this._story) return;

    const isGuest = this._story.name === 'Guest';
    const bookmarkIconClass = this._isBookmarked ? 'fas fa-bookmark' : 'far fa-bookmark';
    
    this.dataset.storyId = this._story.id;
    
    this.innerHTML = `
        <article class="story-item" data-id="${this._story.id}" style="opacity: 0; transform: translateY(20px);">
        <img 
            src="${this._story.photoUrl}" 
            alt="Story by ${this._story.name}" 
            loading="lazy"
            class="story-item__bg-image"
        >
        <div class="story-item__header">
            <h3 class="story-item__title">
            ${isGuest ? '<i class="fas fa-user-secret guest-icon"></i>' : ''} ${this._story.name}
            </h3>
            <button class="bookmark-btn" data-id="${this._story.id}" aria-label="${this._isBookmarked ? 'Remove from bookmarks' : 'Add to bookmarks'}">
              <i class="${bookmarkIconClass}"></i>
            </button>
            <p class="story-item__location">
            ${this._story.lat && this._story.lon ? 
                `<i class="fas fa-map-marker-alt"></i> <span class="location-name">Loading location...</span>` : 
                ''}
            </p>
        </div>
        <div class="story-item__footer">
            <div class="story-item__description-container">
              <p class="story-item__description">
              ${this._story.description.length > 50 
                  ? `${this._story.description.substring(0, 50)}...` 
                  : this._story.description}
              </p>
            </div>
            <p class="story-item__date">${showFormattedDate(this._story.createdAt)}</p>
            <a href="#/stories/${this._story.id}" class="story-item__detail-btn">
              <i class="fas fa-eye"></i>Details
            </a>
        </div>
        </article>
    `;
    
    const card = this.querySelector('.story-item');
    if (card) {
      AnimationHelper.animateStoryCard(card);
    }

    this._initBookmarkButton();
  }
  
  _initBookmarkButton() {
    const bookmarkBtn = this.querySelector('.bookmark-btn');
    if (bookmarkBtn) {
      bookmarkBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        
        try {
          if (this._isBookmarked) {
            await this._presenter.removeBookmark(this._story.id);
            this._isBookmarked = false;
            showAlert('Story removed from bookmarks', 'success');
            
            this.dispatchEvent(new CustomEvent('bookmark:removed', {
              bubbles: true,
              detail: { storyId: this._story.id }
            }));
          } else {
            await this._presenter.addBookmark(this._story);
            this._isBookmarked = true;
            showAlert('Story added to bookmarks', 'success');
          }
          
          const icon = bookmarkBtn.querySelector('i');
          icon.className = this._isBookmarked ? 'fas fa-bookmark' : 'far fa-bookmark';
          bookmarkBtn.setAttribute('aria-label', this._isBookmarked ? 'Remove from bookmarks' : 'Add to bookmarks');
        } catch (error) {
          console.error('Error toggling bookmark:', error);
          showAlert('Failed to update bookmark', 'error');
        }
      });
    }
  }
  
  async _fetchLocationName() {
    if (!this._story || !this._story.lat || !this._story.lon) return;
    
    try {
      const locationName = await this._presenter.getLocationName(this._story.lat, this._story.lon);
      const locationElement = this.querySelector('.location-name');
      if (locationElement) {
        locationElement.textContent = locationName;
      }
    } catch (error) {
      console.error('Error fetching location name:', error);
      const locationElement = this.querySelector('.location-name');
      if (locationElement) {
        locationElement.textContent = `${this._story.lat.toFixed(4)}, ${this._story.lon.toFixed(4)}`;
      }
    }
  }
  
  updateBookmarkStatus(isBookmarked) {
    this._isBookmarked = isBookmarked;
    const bookmarkBtn = this.querySelector('.bookmark-btn');
    if (bookmarkBtn) {
      const icon = bookmarkBtn.querySelector('i');
      icon.className = this._isBookmarked ? 'fas fa-bookmark' : 'far fa-bookmark';
      bookmarkBtn.setAttribute('aria-label', this._isBookmarked ? 'Remove from bookmarks' : 'Add to bookmarks');
    }
  }
  
  updateLocationName(locationName) {
    const locationElement = this.querySelector('.location-name');
    if (locationElement) {
      locationElement.textContent = locationName;
    }
  }
}

customElements.define('story-card', StoryCardHelper);

export default StoryCardHelper;