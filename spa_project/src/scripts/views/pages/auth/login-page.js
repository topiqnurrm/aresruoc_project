import AuthFormHelper from '../../../utils/auth-form-helper';
import LoginPresenter from '../../../presenters/auth/login-presenter';

export default class LoginPage {
  constructor() {
    this.authForm = null;
    this.presenter = new LoginPresenter(this);
  }
  
  async render() {
    return `
        <section class="auth-form">
          <h1 class="auth-form__title">LOGIN</h1>
          <form id="login-form" class="form">
            <div class="form-group form-group--icon">
              <label for="email">Email</label>
              <i class="fas fa-envelope"></i>
              <input 
                type="email" 
                id="email" 
                name="email" 
                class="form-control" 
                placeholder="Enter your email"
                required
                aria-required="true"
              >
            </div>
            
            <div class="form-group form-group--icon">
              <label for="password">Password</label>
              <i class="fas fa-lock"></i>
              <input 
                type="password" 
                id="password" 
                name="password" 
                class="form-control" 
                placeholder="Enter your password"
                required
                aria-required="true"
              >
            </div>
            
            <button type="submit" class="submit-button">
              <i class="fas fa-sign-in-alt"></i> Login
            </button>
          </form>
          
          <p class="auth-form__footer">
            Don't have an account? <a href="#/register">Register here</a>
          </p>
        </section>
    `;
  }

  async afterRender() {
    this.authForm = new AuthFormHelper('login-form', this.presenter.handleLogin.bind(this.presenter));
    this.authForm.init();
  }
}