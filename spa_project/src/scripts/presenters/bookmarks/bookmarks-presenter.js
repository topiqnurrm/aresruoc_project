import StoryIDB from '../../data/idb';
import ErrorHandler from '../../utils/error-handler';

class BookmarksPresenter {
  constructor(view) {
    this._view = view;
    this._storyIDB = StoryIDB;
  }

  async loadBookmarks() {
    try {
      const bookmarks = await this._storyIDB.getAllBookmarks();
      this._view.displayBookmarks(bookmarks);
      return bookmarks;
    } catch (error) {
      ErrorHandler.handleApiError(error, {
        context: 'Loading bookmarks',
        defaultMessage: 'Failed to retrieve your bookmarks'
      });
      return [];
    }
  }

  /**
   * 
   * @param {string} storyId - ID cerita yang bookmark-nya dihapus
   */
  async onBookmarkRemoved(storyId) {
    try {
      const isStillBookmarked = await this._storyIDB.isBookmarked(storyId);
      
      if (isStillBookmarked) {
        await this._storyIDB.removeBookmark(storyId);
      }
    } catch (error) {
      console.error('Error confirming bookmark removal:', error);
    }
  }
}

export default BookmarksPresenter;