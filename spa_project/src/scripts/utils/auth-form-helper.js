import { showAlert } from '../utils';

class AuthFormHelper {
  constructor(formId, submitCallback) {
    this.formId = formId;
    this.submitCallback = submitCallback;
    this.form = null;
    this.submitButton = null;
    
    this._handleSubmit = this._handleSubmit.bind(this);
  }
  
  init() {
    this.form = document.getElementById(this.formId);
    if (!this.form) {
      console.error(`Form with ID "${this.formId}" not found`);
      return false;
    }
    
    this.submitButton = this.form.querySelector('button[type="submit"]');
    this.form.addEventListener('submit', this._handleSubmit);
    return true;
  }
  
  async _handleSubmit(e) {
    e.preventDefault();
    
    const formData = this._getFormData();
    if (!this._validateFormData(formData)) {
      return;
    }
    
    const originalButtonContent = this.submitButton.innerHTML;
    this.submitButton.classList.add('loading');
    this.submitButton.innerHTML = 'Logging in... <i class="fas fa-spinner fa-spin"></i>';
    this.submitButton.disabled = true;
    
    try {
      await this.submitCallback(formData);
    } catch (error) {
      console.error('Form submission error:', error);
      showAlert(error.message || 'An error occurred. Please try again.', 'error');
    } finally {
      this.submitButton.classList.remove('loading');
      this.submitButton.innerHTML = originalButtonContent;
      this.submitButton.disabled = false;
    }
  }
  
  _getFormData() {
    const data = {};
    const formElements = this.form.elements;
    
    for (let i = 0; i < formElements.length; i++) {
      const element = formElements[i];
      if (element.name && element.name !== '') {
        data[element.name] = element.value;
      }
    }
    
    return data;
  }
  
  _validateFormData(data) {
    if (data.email && !this._isValidEmail(data.email)) {
      showAlert('Please enter a valid email address', 'error');
      return false;
    }
    
    if (data.password && data.password.length < 8) {
      showAlert('Password must be at least 8 characters long', 'error');
      return false;
    }
    
    return true;
  }
  
  _isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }
}

export default AuthFormHelper;