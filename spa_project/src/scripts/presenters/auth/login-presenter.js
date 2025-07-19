import StoryApi from '../../data/api';
import { showAlert } from '../../utils';

export default class LoginPresenter {
  constructor(view) {
    this.view = view;
    this.api = StoryApi;
  }

  async handleLogin(formData) {
    try {
      const { loginResult } = await this.api.login({
        email: formData.email,
        password: formData.password
      });
      
      localStorage.setItem('token', loginResult.token);
      localStorage.setItem('name', loginResult.name);
      document.dispatchEvent(new Event('authChanged'));
      showAlert('Login successful!');
      window.location.hash = '#/';
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  }
}