import '../styles/main.css';
import App from './views/app';
import NotificationHelper from './utils/notification-helper';
import { Workbox } from 'workbox-window';
import { showAlert } from './utils';
import NetworkStatus from './utils/network-status';

function createInstallButton() {
  const button = document.createElement('button');
  button.id = 'installButton';
  button.className = 'install-button';
  button.innerHTML = '<i class="fas fa-download"></i> Install App';
  button.style.display = 'none';

  button.addEventListener('click', async () => {
    if (!window.deferredPrompt) {
      showAlert('Installation not available', 'error');
      return;
    }
    
    window.deferredPrompt.prompt();
    const choiceResult = await window.deferredPrompt.userChoice;
    
    if (choiceResult.outcome === 'accepted') {
      showAlert('App installed successfully!', 'success');
    } else {
      showAlert('App installation cancelled', 'info');
    }
    
    window.deferredPrompt = null;
    button.style.display = 'none';
  });

  document.body.appendChild(button);
  return button;
}

async function initializeApp() {
  try {
    if ('serviceWorker' in navigator) {
      const wb = new Workbox('/service-worker.js', { scope: '/' });

      wb.addEventListener('installed', (event) => {
        if (event.isUpdate) {
          console.log('App updated');
          showAlert('App Updated!', 'info');
        } else {
          console.log('App installed and cached for offline use');
          if (window.matchMedia('(display-mode: browser)').matches) {
            window.dispatchEvent(new Event('showInstallPrompt'));
          }
        }
      });

      wb.addEventListener('activated', (event) => {
        if (!event.isUpdate) {
          console.log('Service worker activated for the first time');
        }
      });

      wb.addEventListener('waiting', (event) => {
        showAlert('New version available! Refresh to update.', 'info');
      });

      try {
        await wb.register();
        console.log('Service worker registered');
        
        const token = localStorage.getItem('token');
        if (token) {
          await NotificationHelper.initPushNotifications();
        }
      } catch (error) {
        console.error('Service worker registration failed:', error);
      }
    }

    const contentElement = document.getElementById('main-content');
    if (!contentElement) {
      throw new Error('Main content element not found');
    }

    const app = new App({
      content: contentElement,
      drawerButton: document.querySelector('#drawer-button'),
      navigationDrawer: document.querySelector('#navigation-drawer'),
    });

    const token = localStorage.getItem('token');
    const currentHash = window.location.hash;

    if (!currentHash && !token) {
      window.location.hash = '#/guest';
    }

    await app.renderPage();

    window.addEventListener('hashchange', () => app.renderPage());
    

    const installButton = createInstallButton();
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      window.deferredPrompt = e;
      installButton.style.display = 'block';
    });

    window.addEventListener('appinstalled', () => {
      console.log('PWA installed successfully');
      installButton.style.display = 'none';
      window.deferredPrompt = null;
    });

  } catch (error) {
    console.error('Application initialization failed:', error);
    
    const errorDisplay = document.createElement('div');
    errorDisplay.style.padding = '20px';
    errorDisplay.style.color = 'red';
    errorDisplay.textContent = 'Failed to initialize application. Please refresh the page.';
    
    const mainContent = document.getElementById('main-content') || document.body;
    mainContent.innerHTML = '';
    mainContent.appendChild(errorDisplay);
  }
}

document.addEventListener('DOMContentLoaded', initializeApp);