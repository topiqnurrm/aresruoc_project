import { Link } from 'react-router-dom';
import './Navbar.css';
import { useEffect, useState } from 'react';

const Navbar = () => {
    const [click, setClick] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [showDropdown, setShowDropdown] = useState(false);

    const handleClick = () => setClick(!click);
    
    const handleLogout = () => {
        sessionStorage.removeItem("authtoken");
        sessionStorage.removeItem("name");
        sessionStorage.removeItem("email");
        sessionStorage.removeItem("phone");
        // remove email phone
        localStorage.removeItem("doctorData");
        setIsLoggedIn(false);
        // setUsername("");
       
        // Remove the reviewFormData from local storage
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key.startsWith("reviewFormData_")) {
            localStorage.removeItem(key);
          }
        }
        setEmail('');
        setUsername('');
        window.location.reload();
    }
    const handleDropdown = () => {
      setShowDropdown(!showDropdown);
    }
    
    useEffect(() => {
        const token = sessionStorage.getItem("authtoken");
        const email = sessionStorage.getItem("email");
        const name = email?  email.split('@')[0] : "";
        if (token) {
            setIsLoggedIn(true);
            setUsername(name);
            setEmail(email);
        }
    }, []);

    return (
        <div>
            <nav>
                <div className="nav__logo">
                    <Link to="/">
                        StayHealthy 
                        <i style={{color:'#000000',fontSize:'2rem'}} className="fa fa-user-md"></i>
                        
                    </Link>
                    <span>.</span>
                </div>
                <div className="nav__icon" onClick={handleClick}>
                    {/* <i className="fa fa-times fa fa-bars"></i> */}
                </div>

                <ul className="nav__links active">
                    <li className="link">
                        <Link to="/">Home</Link>
                    </li>
                    <li className="link">
                        <Link to="/instant-consultation">Appointments</Link>
                    </li>
                    {isLoggedIn? (
                        <li className='welcome-user'>
                            <h3 > Welcome, {username}</h3>
                            <ul className="dropdown-menu">
                                <li><a href="/profile">Your Profile</a></li>
                                <li><a href="/reports">Your Reports</a></li>
                            </ul>
                        </li>):
                    <li className="link">
                        <Link to="/signup">
                            <button className="btn1">Sign Up</button>
                        </Link>
                    </li>}
                    <li className="link">
                        {isLoggedIn ? <Link to="#"><button className="btn2" onClick={handleLogout}>Logout</button></Link>:<Link to="/login"><button className="btn1">Login</button></Link>}
                        
                    </li>
                </ul>
            </nav>
        </div>
    );
}

export default Navbar;