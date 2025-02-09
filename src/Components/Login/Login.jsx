import { Link, useNavigate } from 'react-router-dom';
import './Login.css';
import { useState } from 'react';
import { API_URL } from '../../../config';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const navigate = useNavigate();

  const login = async (e) => {
    e.preventDefault();
    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        password,
      }),
    });

    const json = await response.json();
    console.log(json);
    if (json.authtoken) {
      sessionStorage.setItem('authtoken', json.authtoken);
      sessionStorage.setItem('email', email);
      
      navigate('/');
      window.location.reload();
    } else {
      if (json.error) {
        alert(json.error);
      }
    }

  };
    return (
        <div className="container">
    
        <div className="login-grid">
       
          <div className="login-text">
            <h2>Login</h2>
          </div>
         
          <div className="login-text">
            Are you a new member? <span><Link to='/signup' style={{color: '#2190FF'}}> Sign Up Here</Link></span>
          </div>
          <br />
        
          <div className="login-form">
            <form onSubmit={login}>
              
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input type="email" name="email" id="email" className="form-control" 
                placeholder="Enter your email" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                aria-describedby="helpId" />
                
              </div>
              
              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  type="password"
                  name="password"
                  id="password"
                  className="form-control"
                  placeholder="Enter your password"
                  aria-describedby="helpId"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                
              </div>
             
              <div className="btn-group">
                <button type="submit" className="btn btn-primary mb-2 mr-1 waves-effect waves-light"
                >Login</button> 
                <button type="reset" className="btn btn-danger mb-2 waves-effect waves-light"
                    onClick={() => {
                      setEmail('');
                      setPassword('');
                    }}
                    >Reset
                </button>
              </div>
              <br />
             
              <div className="login-text">
                Forgot Password?
              </div>
            </form>
          </div>
        </div>
      </div>
    )};

    export default Login