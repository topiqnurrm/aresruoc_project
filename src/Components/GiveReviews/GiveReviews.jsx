import { useState } from "react";
import { useParams } from "react-router-dom"


export default function GiveReviews (){
    const { id } = useParams();
    const [showForm, setShowForm] = useState(false);
    const [submittedMessage, setSubmittedMessage] = useState('');
    const [showWarning, setShowWarning] = useState(true);
    const [formData, setFormData] = useState({
        id : id,
        name: '',
        review: '',
        rating: 0
    });

    const handlaeButtonClick = () =>{
        setShowForm(true);
    };

    const handleChange = (e) =>{
        setFormData({...formData, [e.target.name]: e.target.value});
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setSubmittedMessage(formData);
        setFormData({
          name: '',
          review: '',
          rating: 0
        });

    if (formData.name && formData.review && formData.rating > 0) {
        setShowWarning(false);
      } else {
        setShowWarning(true);
      }
    };

    return(
        <div>
            <h2>Form with Message</h2>
            {!showForm ? (
                <button onClick={handlaeButtonClick}>Open Form</button>
            ):(
                <form onSubmit={handleSubmit}>
                    <h2>Give your Feedback</h2>
                    {showWarning && <p className="warning"> Please fill out all fields.</p> }
                    <div>
                        <label htmlFor="name">Name:</label>
                        <input type="text" id='name' name='name' value={formData.name} onChange={handleChange}/>
                    </div>
                    <div>
                        <label htmlFor="review">Review:</label>
                        <textarea type="text" id="review" name="review" value={formData.review} onChange={handleChange}/>
                    </div>
                    <div>
                        <label htmlFor="rating">Rating:{}</label>
                    </div>
                    <button type="submit">Submit</button>
                </form>
            )}
            {submittedMessage &&(
                <div>
                    <h3>Submitted Message:</h3>
                    <p>{submittedMessage}</p>
                </div>
            )}
        </div>
    );
}