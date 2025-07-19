class AnimationHelper {
  /**
   * 
   * @param {HTMLElement} element - The DOM element to animate
   * @param {Object} options - Animation options
   * @param {number} options.delay - Delay before animation starts (ms)
   * @param {string} options.duration - CSS duration value (e.g., '0.5s')
   * @param {string} options.easing - CSS easing function (e.g., 'ease')
   * @param {string} options.opacity - Target opacity value
   * @param {string} options.transform - Target transform value
   * @param {Function} options.callback - Optional callback after animation completes
   */
  static animateElement(element, options = {}) {
    if (!element) return;
    
    const {
      delay = 0,
      duration = '0.5s',
      easing = 'ease',
      opacity = '1',
      transform = 'none',
      callback = null
    } = options;
    
    setTimeout(() => {
      element.style.transition = `opacity ${duration} ${easing}, transform ${duration} ${easing}`;
      element.style.opacity = opacity;
      element.style.transform = transform;
      
      if (callback) {
        element.addEventListener('transitionend', () => {
          callback(element);
        }, { once: true });
      }
    }, delay);
  }
  
  /**
   * Animates multiple form elements in sequence
   * @param {NodeList|Array} elements - Form elements to animate
   * @param {number} baseDelay - Initial delay before animations start
   * @param {number} staggerDelay - Delay between each element's animation
   */
  static animateFormElements(elements, baseDelay = 100, staggerDelay = 150) {
    if (!elements || !elements.length) return;
    
    Array.from(elements).forEach((element, index) => {
      this.animateElement(element, {
        delay: baseDelay + (index * staggerDelay),
        transform: 'translateY(0)'
      });
    });
  }
  
  /**
   * 
   * @param {Object} elements - Object containing story detail elements
   */
  static animateStoryDetail(elements) {
    const { header, imageContainer, description, map } = elements;
    
    if (header) {
      this.animateElement(header, {
        delay: 100,
        transform: 'translateY(0)'
      });
    }
    
    if (imageContainer) {
      this.animateElement(imageContainer, {
        delay: 200,
        transform: 'scale(1)'
      });
    }
    
    if (description) {
      this.animateElement(description, {
        delay: 300,
        transform: 'translateX(0)'
      });
    }
    
    if (map) {
      this.animateElement(map, {
        delay: 400,
        duration: '0.8s'
      });
    }
  }
  
  /**
   *
   * @param {HTMLElement} card - The story card element
   * @param {number} delay - Delay before animation starts
   */
  static animateStoryCard(card, delay = 100) {
    if (!card) return;
    
    this.animateElement(card, {
      delay,
      transform: 'translateY(0)'
    });
  }
}

export default AnimationHelper;