import RegisterPresenter from '../../../presenters/auth/register.presenter';

export default class RegisterPage {
  constructor() {
    this.presenter = new RegisterPresenter(this);
  }

  async render() {
    return `
        <section class="auth-form">
          <h1 class="auth-form__title">REGISTER</h1>
          <form id="register-form" class="form">
            <div class="form-group form-group--icon">
              <label for="name">Name</label>
              <i class="fas fa-user"></i>
              <input 
                type="text" 
                id="name" 
                name="name" 
                class="form-control" 
                placeholder="Enter your full name"
                required
                aria-required="true"
              >
            </div>
            
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
                placeholder="Enter Your password"
                required
                aria-required="true"
                minlength="8"
              >
            </div>
            
            <button type="submit" class="submit-button">
              <i class="fas fa-user-plus"></i> Register
            </button>
          </form>
          
          <p class="auth-form__footer">
            Already have an account? <a href="#/login">Login here</a>
          </p>
        </section>
    `;
  }

  async afterRender() {
    const form = document.getElementById('register-form');
    const submitButton = form.querySelector('.submit-button');
    
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      submitButton.classList.add('loading');
      submitButton.innerHTML = 'Registering...';
      
      const name = document.getElementById('name').value;
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      
      try {
        await this.presenter.handleRegister({ name, email, password });
      } catch (error) {
        submitButton.classList.remove('loading');
        submitButton.innerHTML = '<i class="fas fa-user-plus"></i> Register';
      }
    });
  }
}