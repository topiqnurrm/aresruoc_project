import { useEffect, useState } from 'react';
import './PatientReport.css'
import { useNavigate } from 'react-router-dom';


export default function PatientReport(){
    const [doctors, setDoctors] = useState([]);

    const navigate = useNavigate();

    const gotoRiview =(id) => {
        navigate(`/giveReview/${id}`);


    }

    useEffect(()=>{
        fetch('https://api.npoint.io/9a5543d36f1460da2f63')
        .then(res => res.json())
        .then(data => {
            setDoctors(data);
        })
    },[]);

    return(
        <>
        <div className='reports'>
            <h2>Reports</h2>
            <table>
                <thead>
                    <tr>
                        <th>Serial Number</th>
                        <th>Doctor Name</th>
                        <th>Doctor Speciality</th>
                        <th>View Report</th>
                        <th>Download Report</th>
                    </tr>
                </thead>
                <tbody>
                {doctors.map((doctor, index) => (
                    <tr key={doctor.name}>
                        <td>{index+1}</td>
                        <td>{doctor.name}</td>
                        <td>{doctor.speciality}</td>
                        <td>
                            <button style={{ marginBottom: '5px' }} onClick={()=>gotoRiview(index+1)}>View Report</button>
                        </td>
                        <td>
                            <button style={{ marginBottom: '5px' }} onClick={()=>gotoRiview(index+1)}>Download Report</button>
                        </td>
                    </tr>
))}
                </tbody>
            </table>
        </div>
        </>
    )
}