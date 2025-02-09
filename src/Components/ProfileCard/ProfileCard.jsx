import { useEffect, useState } from 'react';
import './ProfileCard.css'
import { useNavigate } from 'react-router-dom';
import { API_URL } from '../../../config';


export default function ProfileCard(){
    const [userDetails, setUserDetails] = useState({});
    const [updatedDetails, setUpdatedDetails] = useState({});
    const [editMode, setEditMode] = useState(false);

    const navigate = useNavigate();

    useEffect(()=>{
        const fetchUserProfile = async() => {
            try {
                const authtoken = sessionStorage.getItem("authtoken");
                const email = sessionStorage.getItem("email");
                if(!authtoken){
                    navigate('/login');
                } else{
                    const response = await fetch(`${API_URL}/api/auth/user`,{
                        headers: {
                            "Authorization": `Bearer ${authtoken}`,
                            "Email": email,
                        },
                    });
                    if (response.ok){
                        const user = await response.json();
                        setUserDetails(user);
                        setUpdatedDetails(user);
                    } else{
                        throw new Error("Failed to fetch user profile");
                    }
                }
            } catch (error){
                console.error(error);
            }
        };
        const authtoken = sessionStorage.getItem("authtoken");
        if(!authtoken){
            navigate("/login");
        } else {
            fetchUserProfile();
        }
    }, [navigate]);

    

    const handleEdit = () =>{
        setEditMode(true);
    };

    const handleInputChange = (e) =>{
        setUpdatedDetails({
            ...updatedDetails, 
            [e.target.name]: e.target.value,
        });
    }

    // Function to handle form submission when user saves changes
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const authtoken = sessionStorage.getItem("authtoken");
      const email = sessionStorage.getItem("email"); // Get the email from session storage
      if (!authtoken || !email) {
        navigate("/login");
        return;
      }
      const payload = { ...updatedDetails };
      const response = await fetch(`${API_URL}/api/auth/user`, {
        method: "PUT",
        headers: {
          "Authorization": `Bearer ${authtoken}`,
          "Content-Type": "application/json",
          "Email": email,
        },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        // Update the user details in session storage
        sessionStorage.setItem("name", updatedDetails.name);
        sessionStorage.setItem("phone", updatedDetails.phone);
        setUserDetails(updatedDetails);
        setEditMode(false);
        // Display success message to the user
        alert(`Profile Updated Successfully!`);
        navigate("/");
      } else {
        // Handle error case
        throw new Error("Failed to update profile");
      }
    } catch (error) {
      console.error(error);
      // Handle error case
    }
  };

    return(
        <div className="profile-container">
      {editMode ? (
        <form onSubmit={handleSubmit}>
            <label>
            Name:
            <input
              type="name"
              name="name"
              value={updatedDetails.name}
               // Disable the email field
               onChange={handleInputChange}
            />
          </label>
          <label>
            Email:
            <input
              type="email"
              name="email"
              value={userDetails.email}
              disabled // Disable the email field
            />
          </label>
          <label>
            Phone:
            <input
              type="phone"
              name="phone"
              value={updatedDetails.phone}
               // Disable the email field
               onChange={handleInputChange}
            />
          </label>
          {/* Create similar logic for displaying and editing name and phone from userDetails */}
          <button type="submit">Save</button>
        </form>
      ) : (
        <div className="profile-details">
          <h1>Welcome, {userDetails.name}</h1>
          {/* Implement code to display and allow editing of phone and email similar to above */}
          <button onClick={handleEdit}>Edit</button>
        </div>
      )}
    </div>
    );
}