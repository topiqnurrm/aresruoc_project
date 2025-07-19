import { showAlert } from '../../../utils';
import '../../../utils/story-card-helper';
import '../../pages/story/story-list';
import AnimationHelper from '../../../utils/animation-helper';
import ErrorHandler from '../../../utils/error-handler';
import BookmarksPresenter from '../../../presenters/bookmarks/bookmarks-presenter';

class BookmarksPage {
  constructor() {
    this._presenter = new BookmarksPresenter(this);
  }
  
  async render() {
    return `
      <section class="bookmarks-page">
        <h2 class="page-title">Your Bookmarked Stories</h2>
        
        <div class="bookmarks-container">
          <story-list id="bookmarks-list" class="story-grid"></story-list>
        </div>
      </section>
    `;
  }

  async afterRender() {
    try {
      const bookmarksList = document.querySelector('#bookmarks-list');
      await this._presenter.loadBookmarks();
      
      this._initBookmarkRemovedListener(bookmarksList);
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Loading bookmarks',
        defaultMessage: 'Failed to load your bookmarks'
      });
    }
  }
  
  displayBookmarks(bookmarks) {
    const bookmarksList = document.querySelector('#bookmarks-list');
    
    if (bookmarks.length === 0) {
      this._renderEmptyBookmarks(bookmarksList);
      return;
    }
    
    bookmarksList.stories = bookmarks;
  }
  
  _renderEmptyBookmarks(bookmarksList) {
    const emptyState = document.createElement('div');
    emptyState.className = 'no-bookmarks';
    emptyState.innerHTML = `
      <i class="fas fa-bookmark empty-icon"></i>
      <p>You haven't bookmarked any stories yet</p>
      <a href="#/" class="explore__button">Explore Stories</a>
    `;
    
    bookmarksList.innerHTML = '';
    bookmarksList.appendChild(emptyState);
    
    AnimationHelper.animateElement(emptyState, {
      delay: 100,
      transform: 'scale(1)',
      duration: '0.6s'
    });
  }
  
  _initBookmarkRemovedListener(bookmarksList) {
    document.addEventListener('bookmark:removed', async (event) => {
      if (event.detail && event.detail.storyId) {
        const storyId = event.detail.storyId;
        
        await this._presenter.onBookmarkRemoved(storyId);
        
        const storyCard = bookmarksList.querySelector(`story-card[data-story-id="${storyId}"]`);
        
        if (storyCard) {
          AnimationHelper.animateElement(storyCard, {
            opacity: '0',
            transform: 'translateX(20px)',
            duration: '0.3s',
            callback: () => {
              storyCard.remove();
              
              const remainingCards = bookmarksList.querySelectorAll('story-card');
              if (remainingCards.length === 0) {
                this._renderEmptyBookmarks(bookmarksList);
              }
            }
          });
        }
      }
    });
  }
}

export default BookmarksPage;