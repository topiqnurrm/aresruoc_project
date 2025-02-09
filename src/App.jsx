import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './App.css'
import Landing_Page from './Components/Landing_Page/Landing_Page'
import Login from './Components/Login/Login'
import Sign_Up from './Components/Sign_Up/Sign_Up'
import InstantConsultation from './Components/InstantConsultationBooking/InstantConsultation'
import Notifications from './Components/Notification/Notification'
import ReviewForm from './Components/ReviewForm/ReviewForm'
import GiveReviews from './Components/GiveReviews/GiveReviews'
import ProfileCard from './Components/ProfileCard/ProfileCard'
import PatientReport from './Components/PatientReport/PatientReport'
function App() {

  return (
    <div className="App">
      <BrowserRouter>
        {/* <Navbar /> */}
        <Notifications />
        <Routes>
          <Route path="/" element={<Landing_Page />} />
          <Route path="/login" element={<Login />} />
          <Route path='/signup' element={<Sign_Up />} />
          <Route path='/instant-consultation' element= {<InstantConsultation />} />
          <Route path='/review' element = {<ReviewForm />} />
          <Route path='/giveReview/:id' element = {<GiveReviews />} />
          <Route path='/profile' element = {<ProfileCard />} />
          <Route path='/reports' element = {<PatientReport />}/>
          <Route path='*' element={<h1>404 Not Found</h1>} />
        </Routes>
      </BrowserRouter>
    </div>
  )
}

export default App
