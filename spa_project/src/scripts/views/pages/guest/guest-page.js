export default class GuestPage {
  async render() {
    return `
        <section class="hero">
          <div class="hero__inner">
            <h1 class="hero__title">Share Your Story</h1>
            <p class="hero__tagline">Tell your amazing experiences and inspire others</p>
            <div class="hero__cta">
              <a href="#/add-guest-story" class="hero-button">
                <i class="fas fa-plus"></i> Add Your Story
              </a>
            </div>
          </div>
        </section>
        
        <div class="container">
          <section class="content-section stories-section">
            <h2 class="section-title">Recent Stories</h2>
            <div class="guest-message">
                <div class="guest-card">
                  <i class="fas fa-lock"></i>
                  <h3>Login Required</h3>
                  <p>You need to login first to view stories from our web.</p>
                  <div class="guest-buttons">
                    <a href="#/login" class="guest-button primary">Login</a>
                    <a href="#/register" class="guest-button secondary">Register</a>
                  </div>
                </div>
              </div>
          </section>
        </div>
    `;
  }

  async afterRender() {
    
  }
}