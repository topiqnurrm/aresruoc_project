import StoryApi from '../../data/api';
import { showAlert } from '../../utils';

export default class RegisterPresenter {
  constructor(view) {
    this.view = view;
    this.api = StoryApi;
  }

  async handleRegister(formData) {
    try {
      await this.api.register(formData);
      showAlert('Registration successful! Please login.');
      window.location.hash = '#/login';
    } catch (error) {
      console.error('Registration failed:', error);
      throw error;
    }
  }
}