import { useState } from 'react';
import './AppointmentForm.css';

const AppointmentForm = ({ doctorName, doctorSpeciality, onSubmit }) => {
    const [name, setName] = useState('');
    const [phoneNumber, setPhoneNumber] = useState('');
    const [date, setDate] = useState('');
    const [selectedSlot, setSelectedSlot] = useState(null);

    const timeSlots = [
      "09:00 AM",
      "10:00 AM",
      "11:00 AM",
      "12:00 PM",
      "01:00 PM",
      "02:00 PM",
      "03:00 PM",
      "04:00 PM",
    ];
  
    // const handleSlotSelection = (slot) => {
    //   setSelectedSlot(slot);
    // };
  
    const handleFormSubmit = (e) => {
      e.preventDefault();
      onSubmit({ name, phoneNumber, date, selectedSlot });
      setName('');
      setPhoneNumber('');
    };
  
    return (
      <form onSubmit={handleFormSubmit} className="appointment-form">
        <div className="form-group">
          <label htmlFor="name">Name:</label>
          <input
            type="text"
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="phoneNumber">Phone Number:</label>
          <input
            type="tel"
            id="phoneNumber"
            value={phoneNumber}
            onChange={(e) => setPhoneNumber(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="date">Date of Appoinment:</label>
          <input 
          type="date"
          id="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          required
          /></div>
        <div className="form-group">
      <label htmlFor="selectSlot">Book Time Slot:</label>
      <select
        id="selectSlot"
        value={selectedSlot}
        onChange={(e) => setSelectedSlot(e.target.value)}
        required
      >
        <option value="">Select a time slot</option>
        {timeSlots.map((slot, index) => (
          <option key={index} value={slot}>
            {slot}
          </option>
        ))}
      </select>

      {selectedSlot && (
        <p>
          You have selected: <strong>{selectedSlot}</strong>
        </p>
      )}
    </div>
        <button type="submit">Book Now</button>
      </form>
    );
  };

export default AppointmentForm
