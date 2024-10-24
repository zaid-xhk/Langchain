from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# In-memory storage for appointments (you can use a database in real implementation)
appointments = {}

class AppointmentRequest(BaseModel):
    user_id: str
    doctor: str
    date: str  # format: 'YYYY-MM-DD'
    time: str  # format: 'HH:MM'

@app.post("/book_appointment/")
async def book_appointment(request: AppointmentRequest):
    appointment_id = len(appointments) + 1
    appointment_key = f"{request.user_id}-{appointment_id}"
    
    # Convert date and time to a datetime object
    appointment_datetime = f"{request.date} {request.time}"
    
    # Store the appointment
    appointments[appointment_key] = {
        "user_id": request.user_id,
        "doctor": request.doctor,
        "datetime": appointment_datetime
    }
    
    return {"message": f"Appointment booked with {request.doctor} on {appointment_datetime}", 
            "appointment_id": appointment_id}
  