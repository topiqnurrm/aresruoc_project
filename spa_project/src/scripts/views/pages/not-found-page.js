export default class NotFoundPage {
    async render() {
      return `
          <div class="not-found-container">
            <div class="not-found-content">
              <h1>404 woy</h1>
              <h2>Page Not Found</h2>
              <p>The page you're looking for doesn't exist or has been moved.</p>
              <a href="#/" class="not-found-button">Back to Home</a>
            </div>
          </div>
      `;
    }
  
    async afterRender() {
    }
  }