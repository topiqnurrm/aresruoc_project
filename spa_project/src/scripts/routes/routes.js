import HomePage from '../views/pages/home/home-page';
import GuestPage from '../views/pages/guest/guest-page'; 
import StoryDetailPage from '../views/pages/story/story-detail-page';
import AddStoryPage from '../views/pages/story/add-story-page';
import LoginPage from '../views/pages/auth/login-page';
import RegisterPage from '../views/pages/auth/register-page';
import AddGuestStoryPage from '../views/pages/guest/add-guest-story-page'; 
import NotFoundPage from '../views/pages/not-found-page';
import BookmarksPage from '../views/pages/bookmarks/bookmarks-page';

const routes = {
  '/': new HomePage(),
  '/home': new HomePage(),
  '/guest': new GuestPage(), 
  '/stories/:id': new StoryDetailPage(),
  '/add-story': new AddStoryPage(),
  '/login': new LoginPage(),
  '/register': new RegisterPage(),
  '/add-guest-story': new AddGuestStoryPage(),
  '/bookmarks': new BookmarksPage(),
  '*': new NotFoundPage(),
};

export default routes;