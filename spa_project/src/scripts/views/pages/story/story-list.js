import '../../../utils/story-card-helper.js';

class StoryList extends HTMLElement {
  set stories(stories) {
    this._stories = stories;
    this.render();
  }

  render() {
    if (!this._stories || this._stories.length === 0) {
      this.innerHTML = `
        <div class="no-stories">
          <p>No stories available yet.</p>
        </div>
      `;
      return;
    }
    
    this.innerHTML = '';
    
    this._stories.forEach(story => {
      const storyCard = document.createElement('story-card');
      storyCard.story = story;
      this.appendChild(storyCard);
    });
  }

  renderError(message) {
    this.innerHTML = `
      <div class="error-message">
        <p>${message}</p>
      </div>
    `;
  }
}

customElements.define('story-list', StoryList);