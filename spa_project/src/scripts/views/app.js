import routes from '../routes/routes';
import { getActiveRoute, parseActivePathname } from '../routes/url-parser';
import { showAlert } from '../utils';
import NotFoundPage from './pages/not-found-page';
import NotificationHelper from '../utils/notification-helper';

class App {
  #content = null;
  #drawerButton = null;
  #navigationDrawer = null;
  #currentPageType = null;

  constructor({ navigationDrawer, drawerButton, content }) {
    if (!content) {
      throw new Error('Content element is required');
    }
    this.#content = content;
    this.#drawerButton = drawerButton;
    this.#navigationDrawer = navigationDrawer;

    this.#setupDrawer();
    this.#checkAuth();
    this.#setupAuthLinks();
    this.#setupStorageListener();

    const skipLink = document.querySelector('.skip-link');
    if (skipLink) {
      skipLink.addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('main-content').setAttribute('tabindex', '-1');
        document.getElementById('main-content').focus();
      });
    }
  }

  #setupDrawer() {
    const overlay = document.getElementById('nav-menu-overlay');
    
    this.#drawerButton.addEventListener('click', () => {
      this.#navigationDrawer.classList.toggle('open');
      if (overlay) {
        overlay.classList.toggle('active');
      }
    });

    if (overlay) {
      overlay.addEventListener('click', () => {
        this.#navigationDrawer.classList.remove('open');
        overlay.classList.remove('active');
      });
    }

    document.body.addEventListener('click', (event) => {
      if (
        !this.#navigationDrawer.contains(event.target) &&
        !this.#drawerButton.contains(event.target)
      ) {
        this.#navigationDrawer.classList.remove('open');
        if (overlay) {
          overlay.classList.remove('active');
        }
      }

      this.#navigationDrawer.querySelectorAll('a').forEach((link) => {
        if (link.contains(event.target)) {
          this.#navigationDrawer.classList.remove('open');
          if (overlay) {
            overlay.classList.remove('active');
          }
        }
      });
    });
  }

  #checkAuth() {
    const token = localStorage.getItem('token');
    if (!token && getActiveRoute() === '/add-story') {
      showAlert('Please login to add a story', 'error');
      window.location.hash = '#/login';
    }
  }

  #setupAuthLinks() {
    const authLinks = this.#navigationDrawer.querySelector('#auth-links');
    const token = localStorage.getItem('token');
    const name = localStorage.getItem('name');
    
    if (token) {
      authLinks.innerHTML = `
        <li><a href="#/add-story" class="nav-link">
          <i class="fas fa-plus-circle"></i> Add Story
        </a></li>
        <li><a href="#/bookmarks" class="nav-link">
          <i class="fas fa-bookmark"></i> Bookmarks
        </a></li>
        <li><button id="notification-btn" class="nav-link notification-btn">
          <i class="far fa-bell"></i> Notifications
        </button></li>
        <li><a href="#/profile" class="nav-link user-info">
          <i class="fas fa-user"></i> ${name}
        </a></li>
        <li><a href="#" id="logout-btn" class="nav-link">
          <i class="fas fa-sign-out-alt"></i> Logout
        </a></li>
      `;
      this._initNotificationButton();
      authLinks.querySelector('#logout-btn').addEventListener('click', this.#handleLogout);
    } else {
      authLinks.innerHTML = `
        <li><a href="#/login" class="nav-link">
          <i class="fas fa-sign-in-alt"></i> Login
        </a></li>
      `;
    }
  }
  
  async _initNotificationButton() {
    const notificationBtn = document.getElementById('notification-btn');
    if (!notificationBtn) return;
    
    try {
      const isSubscribed = await NotificationHelper.isSubscribed();
      this._updateNotificationButtonState(notificationBtn, isSubscribed);
      
      notificationBtn.addEventListener('click', async () => {
        try {
          notificationBtn.disabled = true;
          
          const result = await NotificationHelper.togglePushNotification();
          const newSubscriptionState = await NotificationHelper.isSubscribed();
          
          if (result) {
            showAlert(
              newSubscriptionState 
                ? 'Successfully subscribed to notifications' 
                : 'Successfully unsubscribed from notifications',
              'success'
            );
          } else {
            showAlert('Failed to update notification settings', 'error');
          }
          
          this._updateNotificationButtonState(notificationBtn, newSubscriptionState);
        } catch (error) {
          console.error('Error toggling notifications:', error);
          showAlert('Error updating notification settings', 'error');
        } finally {
          notificationBtn.disabled = false;
        }
      });
    } catch (error) {
      console.error('Error initializing notification button:', error);
    }
  }
  
  _updateNotificationButtonState(button, isSubscribed) {
    const icon = button.querySelector('i');
    icon.className = isSubscribed ? 'fas fa-bell' : 'far fa-bell';
    button.title = isSubscribed ? 'Unsubscribe from notifications' : 'Subscribe to notifications';
    button.innerHTML = `${icon.outerHTML} ${isSubscribed ? 'Subscribed' : 'Notifications'}`;
  }

  #handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('name');
    showAlert('Logged out successfully');
    this.#setupAuthLinks();
    window.location.hash = '#/login';
  }

  #setupStorageListener() {
    window.addEventListener('storage', (event) => {
      if (event.key === 'token' || event.key === 'name') {
        this.#setupAuthLinks();
      }
    });
    
    document.addEventListener('authChanged', () => {
      this.#setupAuthLinks();
    });
  }

  #getPageTypeFromUrl(url) {
    if (url === '/' || url === '/home' || url === '/guest') {
      return 'page-home';
    } else if (url.includes('/stories/')) {
      return 'page-detail';
    } else if (url === '/add-story' || url === '/add-guest-story') {
      return 'page-add-story';
    } else if (url === '/login' || url === '/register') {
      return 'page-auth';
    }
    return 'page-default';
  }

  #applyCustomAnimation(fromPageType, toPageType) {
    this.#content.style.animation = 'none';
    
    void this.#content.offsetWidth;
    
    let animationName;
    
    if (fromPageType === 'page-home' && toPageType === 'page-detail') {
      animationName = 'slideInRight';
    } else if (fromPageType === 'page-detail' && toPageType === 'page-home') {
      animationName = 'slideInLeft';
    } else if (toPageType === 'page-add-story') {
      animationName = 'scaleUp';
    } else if (toPageType === 'page-auth') {
      animationName = 'fadeIn';
    } else {
      animationName = 'fadeIn'; 
    }
    
    this.#content.style.animation = `${animationName} 0.5s ease-out forwards`;
  }

  #highlightActiveNavLink(url) {
    const navLinks = this.#navigationDrawer.querySelectorAll('.nav-link');
    navLinks.forEach((link) => {
      if (link.getAttribute('href') === `#${url}`) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }

  async renderPage() {
    try {
      const url = getActiveRoute();
      const page = routes[url] || new NotFoundPage();
  
      if (!page) {
        window.location.hash = '#/';
        return;
      }
  
      const token = localStorage.getItem('token');
      
      if ((url === '/' || url === '/home') && !token) {
        window.location.hash = '#/guest';
        return;
      }
  
      if ((url === '/add-story' || url === '/profile') && !token) {
        showAlert('Please login to access this page', 'error');
        window.location.hash = '#/login';
        return;
      }
      
      this.#highlightActiveNavLink(url);
      
      const previousPageType = this.#currentPageType;
      const newPageType = this.#getPageTypeFromUrl(url);
      this.#currentPageType = newPageType;
  
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      if (document.startViewTransition && !reducedMotion) {
        await document.startViewTransition(async () => {
          this.#content.innerHTML = await page.render();
          
          this.#content.className = newPageType;
          
          if (previousPageType) {
            this.#applyCustomAnimation(previousPageType, newPageType);
          }
          
          await page.afterRender();
        }).finished;
      } else {
        this.#content.innerHTML = await page.render();
        this.#content.className = newPageType;
        
        if (previousPageType) {
          this.#applyCustomAnimation(previousPageType, newPageType);
        }
        
        await page.afterRender();
      }
    } catch (error) {
      console.error('Render error:', error);
      
      this.#content.innerHTML = `
        <div class="error-state">
          <div class="error-icon">
            <i class="fas fa-exclamation-circle"></i>
          </div>
          <h2>Something went wrong</h2>
          <p>${!navigator.onLine ? 'You are offline. Some content may not be available.' : 'Failed to load content.'}</p>
          <button id="error-retry" class="retry-button">Try Again</button>
        </div>
      `;
      
      const retryButton = document.getElementById('error-retry');
      if (retryButton) {
        retryButton.addEventListener('click', () => {
          if (navigator.onLine) {
            showAlert('Retrying...', 'info');
            this.renderPage();
          } else {
            showAlert('Still offline. Your changes will be saved and synced when online.', 'warning');
          }
        });
      }
    }
  }
}

export default App;