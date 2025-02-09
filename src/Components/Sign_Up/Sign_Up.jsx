import { Link, useNavigate } from "react-router-dom";    // Importing Link from react-router-dom
import './Sign_Up.css';                     // Importing Sign_Up.css file
import { useState } from 'react';            // Importing useState from react
import { API_URL } from "../../../config";

const Sign_Up = () => { 
    const [formData, setFormData] = useState({name: '', phone: '', email: '', password: ''});    // Initializing formData state variable with name, phone, email, password
    const [errors, setErrors] = useState({name: '', phone: '', email: '', password: ''});    // Initializing errors state variable with name, phone, email, password
    // const [showError, setShowError] = useState("");    // Initializing showError state variable with empty string
    const navigate = useNavigate();

    const register = async (e) => {    // Defining register function
        e.preventDefault();    // Preventing default form submission
        const response = await fetch(`${API_URL}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: formData.name,
                phone: formData.phone,
                email: formData.email,
                password: formData.password
            }),
        });

        const json = await response.json();

        if (json.authtoken) {
            sessionStorage.setItem('authtoken', json.authtoken);
            sessionStorage.setItem('name', formData.name);
            sessionStorage.setItem('email', formData.email);
            sessionStorage.setItem('phone', formData.phone);

            navigate('/');
            window.location.reload();
        } else {
            if (json.error) {
                setErrors({name: '', phone: '', email: '', password: ''});
                for (const error of json.error) {
                    setErrors((prev) => {
                        return {
                            ...prev,
                            [error.param]: error.msg
                        }
                    })
                }
            } else {
                setErrors(json.error)
            }
        }
    };

    const updateData = (prevData, e) =>{
        return {
            ...prevData,
            [e.target.name]: e.target.value
        }
    }
    return (
        <div className="container" style={{marginTop: '5%'}}> 
        <div className="signup-grid"> 
            <div className="signup-text"> 
                <h1>Sign Up</h1>
            </div>
            <div className="signup-text1" style={{textAlign: 'left'}}> 
                Already a member? <span><Link to="/login" style={{color: '#2190FF'}}> Login</Link></span>
            </div>
            <div className="signup-form"> 
                <form onSubmit={register}> 

                    <div className="form-group"> 
                        <label htmlFor="name">Name</label> 
                        <input 
                            type="text" 
                            name="name" 
                            id="name" 
                            required 
                            className="form-control" 
                            placeholder="Enter your name" 
                            aria-describedby="helpId" 
                            value={formData.name}
                            onChange={(e) => setFormData(updateData(formData, e))}
                            /> 
                            {errors.name && <div className="text-danger">{errors.name}</div>}
                    </div>

                    <div className="form-group"> 
                        <label htmlFor="phone">Phone</label> 
                        <input 
                            type="tel" 
                            name="phone" 
                            id="phone" 
                            required 
                            className="form-control" 
                            placeholder="Enter your phone number" 
                            aria-describedby="helpId" 
                            value={formData.phone}
                            onChange={(e) => setFormData(updateData(formData, e))}
                            /> 
                            {errors.phone && <div className="text-danger">{errors.phone}</div>}
                    </div>

                    <div className="form-group"> 
                        <label htmlFor="email">Email</label>                         
                        <input 
                            type="email" 
                            name="email" 
                            id="email" 
                            required 
                            className="form-control" 
                            placeholder="Enter your email" 
                            aria-describedby="helpId" 
                            value={formData.email}
                            onChange={(e) => setFormData(updateData(formData, e))}
                            /> 
                            {errors.email && <div className="text-danger">{errors.email}</div>}
                    </div>

                    <div className="form-group"> 
                        <label htmlFor="password">Password</label> 
                        <input 
                            name="password" 
                            type="password"
                            id="password"   
                            required 
                            className="form-control" 
                            placeholder="Enter your password" 
                            aria-describedby="helpId"
                            value={formData.password}
                            onChange={(e) => setFormData(updateData(formData, e))}
                             /> 
                            {errors.password && <div className="text-danger">{errors.password}</div>}
                    </div>


                    <div className="btn-group"> 
                        {/* {showError && <div className="text-danger">{showError}</div>} */}
                        <button type="submit" className="btn btn-primary mb-2 mr-1 waves-effect waves-light">Submit</button> 
                        <button 
                            type="reset" 
                            className="btn btn-danger mb-2 waves-effect waves-light"
                            onClick={() => {
                                setFormData({name: '', phone: '', email: '', password: ''});
                                setErrors({name: '', phone: '', email: '', password: ''});
                            }}
                            >Reset</button> 
                    </div>
                </form> 
            </div>
        </div>
    </div>
    )
}

export default Sign_Up;    // Exporting Sign_Up component