import { showAlert } from './index';

/**
 * A centralized error handling service for consistent error management
 * throughout the application.
 */
class ErrorHandler {
  /**
   * Handles API errors with appropriate user feedback and logging
   * @param {Error} error - The error object
   * @param {Object} options - Error handling options
   * @param {string} options.context - The context where the error occurred
   * @param {string} options.defaultMessage - Default user-friendly message
   * @param {boolean} options.showNotification - Whether to show a notification to the user
   * @returns {Error} The original error for further handling if needed
   */
  static handleApiError(error, { 
    context = 'API',
    defaultMessage = 'An error occurred. Please try again later.',
    showNotification = true
  } = {}) {
    console.error(`[${context}] Error:`, error);
    
    let userMessage = defaultMessage;
    
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      userMessage = 'Network error. Please check your connection and try again.';
    } else if (error.message.includes('401') || error.message.includes('Unauthorized')) {
      userMessage = 'Your session has expired. Please login again.';
    } else if (error.message.includes('403') || error.message.includes('Forbidden')) {
      userMessage = 'You don\'t have permission to perform this action.';
    } else if (error.message.includes('404') || error.message.includes('Not found')) {
      userMessage = 'The requested resource was not found.';
    } else if (error.message.includes('timeout') || error.message.includes('timed out')) {
      userMessage = 'Request timed out. Please try again.';
    }
    
    if (showNotification) {
      showAlert(userMessage, 'error');
    }
    
    return error;
  }
  
  /**
   * Handles errors in form submissions
   * @param {Error} error - The error object
   * @param {Object} options - Error handling options
   * @param {HTMLElement} options.form - The form element
   * @param {HTMLElement} options.submitButton - The form's submit button
   * @param {string} options.context - Context description
   * @param {string} options.defaultMessage - Default user-friendly message
   * @returns {Error} The original error for further handling if needed
   */
  static handleFormError(error, {
    form = null,
    submitButton = null,
    context = 'Form submission',
    defaultMessage = 'There was a problem submitting the form. Please try again.'
  } = {}) {
    this.handleApiError(error, { context, defaultMessage });
    
    if (submitButton) {
      submitButton.disabled = false;
      if (submitButton.classList.contains('loading')) {
        submitButton.classList.remove('loading');
        submitButton.innerHTML = submitButton.getAttribute('data-original-text') || 'Submit';
      }
    }
    
    return error;
  }
  
  /**
   * Handles offline-related errors
   * @param {Error} error - The error object
   * @param {Object} options - Error handling options
   * @param {Function} options.offlineAction - Function to execute for offline handling
   * @param {Function} options.onlineAction - Function to execute if we're online but error occurred
   * @returns {Error} The original error for further handling if needed
   */
  static handleOfflineError(error, {
    offlineAction = null,
    onlineAction = null
  } = {}) {
    const isOfflineError = !navigator.onLine || 
      error.message.includes('Failed to fetch') || 
      error.message.includes('NetworkError');
    
    if (isOfflineError) {
      showAlert('You are offline. Some features may not be available.', 'warning');
      
      if (typeof offlineAction === 'function') {
        offlineAction(error);
      }
    } else {
      this.handleApiError(error);
      
      if (typeof onlineAction === 'function') {
        onlineAction(error);
      }
    }
    
    return error;
  }
}

export default ErrorHandler;