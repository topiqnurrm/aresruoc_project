import { showFormattedDate, initMap, showAlert } from '../../../utils';
import { parseActivePathname } from '../../../routes/url-parser';
import StoryDetailPresenter from '../../../presenters/story/story-detail-presenter';
import AnimationHelper from '../../../utils/animation-helper';
import ErrorHandler from '../../../utils/error-handler';

export default class StoryDetailPage {
  constructor() {
    this.presenter = new StoryDetailPresenter(this);
    this._story = null;
    this._isBookmarked = false;
  }

  async render() {
    return `
        <section class="story-detail">
          <div class="story-detail__container">
            <div class="story-detail__header" style="opacity: 0; transform: translateY(-20px);">
              <div class="story-detail__title-area">
                <h1 class="story-detail__title">Loading Story...</h1>
                <div class="story-detail__meta"></div>
              </div>
              <div class="story-detail__actions">
                <button id="bookmark-button" class="button bookmark-icon-button" aria-label="Bookmark this story">
                  <i class="far fa-bookmark"></i>
                </button>
              </div>
            </div>
            <div class="story-detail__content">
              <div class="story-detail__image-container" style="opacity: 0; transform: scale(0.9);">
                <img class="story-detail__image skeleton" alt="Story image" />
              </div>
              <div class="story-detail__description" style="opacity: 0; transform: translateX(20px);"></div>
            </div>
            <div id="story-map" class="story-detail__map" style="opacity: 0;"></div>
          </div>
        </section>
    `;
  }

  async afterRender() {
    const { id } = parseActivePathname();
    if (!id) {
      window.location.hash = '#/';
      return;
    }
    
    try {
      await this.presenter.loadStoryDetail(id);
      
      this._setupBookmarkButton();
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Loading story detail',
        showNotification: false
      });
    }
  }
  
  _setupBookmarkButton() {
    const bookmarkButton = document.getElementById('bookmark-button');
    if (!bookmarkButton) return;
    
    bookmarkButton.addEventListener('click', async () => {
      try {
        if (this._isBookmarked) {
          await this.presenter.removeBookmark(this._story.id);
          this._isBookmarked = false;
          showAlert('Story removed from bookmarks', 'info');
        } else {
          await this.presenter.addBookmark(this._story);
          this._isBookmarked = true;
          showAlert('Story added to bookmarks', 'success');
        }
        this._updateBookmarkButton();
      } catch (error) {
        ErrorHandler.handleApiError(error, {
          context: 'Updating bookmark',
          defaultMessage: 'Failed to update bookmark'
        });
      }
    });
  }

  _updateBookmarkButton() {
    const bookmarkButton = document.getElementById('bookmark-button');
    if (!bookmarkButton) return;
    
    const icon = bookmarkButton.querySelector('i');
    icon.className = this._isBookmarked ? 'fas fa-bookmark' : 'far fa-bookmark';
  }

  updateStoryDetail(story, isBookmarked) {
    this._story = story;
    this._isBookmarked = isBookmarked;
    this._updateBookmarkButton();
    
    const isGuest = story.name === 'Guest';

    const titleElement = document.querySelector('.story-detail__title');
    titleElement.innerHTML = `
      ${isGuest ? 
        '<i class="fas fa-user-secret"></i>' : 
        '<i class="fas fa-user"></i>'} 
      ${story.name || 'Untitled Story'}
    `;
    const metaElement = document.querySelector('.story-detail__meta');
    metaElement.innerHTML = `
      <span class="story-detail__author">By ${story.name || 'Anonymous'}</span>
      <span class="story-detail__date">${showFormattedDate(story.createdAt)}</span>
    `;

    const imageElement = document.querySelector('.story-detail__image');
    if (story.photoUrl) {
      imageElement.src = story.photoUrl;
      imageElement.alt = `Story by ${story.name || 'Anonymous'}`;
      imageElement.classList.remove('skeleton');
    } else {
      imageElement.src = '/public/images/default-story.jpg';
      imageElement.alt = 'Default story image';
    }

    const descriptionElement = document.querySelector('.story-detail__description');
    descriptionElement.textContent = story.description || 'No description provided';

    const mapElement = document.getElementById('story-map');
    if (story.lat && story.lon) {
      initMap(
        'story-map',
        story.lat,
        story.lon,
        `<b>${story.name || 'Story Location'}</b><br>${story.description || ''}`
      );
    } else {
      mapElement.innerHTML = `
        <p class="no-location">No location data available for this story</p>
      `;
    }
  }

  animateElements() {
    const header = document.querySelector('.story-detail__header');
    const imageContainer = document.querySelector('.story-detail__image-container');
    const description = document.querySelector('.story-detail__description');
    const map = document.getElementById('story-map');
    
    AnimationHelper.animateStoryDetail({
      header,
      imageContainer,
      description,
      map
    });
  }

  showErrorState() {
    const container = document.querySelector('.story-detail__container');
    container.innerHTML = `
      <div class="error-state">
        <h2>Failed to Load Story</h2>
        <p>We couldn't load the story details. Please try again later.</p>
        <a href="#/" class="back-button">Back to Home</a>
      </div>
    `;
  }
}