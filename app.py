from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Float, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import base64
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
# from picamera2 import Picamera2
import cv2
# Initialize FastAPI
app = FastAPI()
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Labels for plant diseases
labels = ["Healthy", "Gray_leaf_spot", "Common_Rust", "Blight_leaves","no_predict"]

# Load TFLite model
try:
    interpreter = tflite.Interpreter(model_path="converted_tflite/vww_96_grayscale_quantized.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError("Failed to load TFLite model: {}".format(str(e)))

# Input and output details for TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
CONFIDENCE_THRESHOLD = 0.80

# Database setup
engine = create_engine("sqlite:///sensor_data.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model for storing sensor data
class SensorDataModel(Base):
    __tablename__ = "sensor_data"
    id = Column(Integer, primary_key=True, index=True)
    serial_number = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    nitrogen = Column(Float)
    potassium = Column(Float)
    moisture = Column(Float)
    eclec = Column(Float)
    phosphorus = Column(Float)
    soilPH = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic model for incoming sensor data
class SensorData(BaseModel):
    serial_number: str
    temperature: float
    humidity: float
    nitrogen: float
    potassium: float
    moisture: float
    eclec: float
    phosphorus: float
    soilPH: float

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global variable to manage data saving state
@app.on_event("startup")
async def startup_event():
    app.state.data_saving_enabled = True

@app.post("/toggle_data_saving")
async def toggle_data_saving():
    app.state.data_saving_enabled = not app.state.data_saving_enabled
    return {"data_saving_enabled": app.state.data_saving_enabled}

# Preprocess frame function
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (96, 96))
    input_data = np.expand_dims(resized_frame, axis=2)
    input_data = np.expand_dims(input_data, axis=0)
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32) / 255.0
    return input_data

# Run model function
def run_model_on_frame(frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_idx = np.argmax(output_data[0])
    confidence = output_data[0][predicted_idx]
    return labels[predicted_idx], confidence

# Prediction response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    image: str

# Serve the HTML template
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_model=PredictionResponse)
def predict():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise HTTPException(status_code=500, detail="Camera not accessible.")
    try:
        ret, frame = camera.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to capture image from camera.")
        prediction, confidence = run_model_on_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
    finally:
        camera.release()
    
    if confidence < CONFIDENCE_THRESHOLD:
        return {"prediction": "No Prediction", "confidence": confidence, "image": image_base64}
    
    return {"prediction": prediction, "confidence": confidence, "image": image_base64}
# @app.get("/predict", response_model=PredictionResponse)
# def predict():
#     try:
#         # Initialize and configure picamera2
#         picam2 = Picamera2()
#         picam2.start()
#         frame = picam2.capture_array()
#         picam2.stop()
#         # Capture the frame
       
#         # Run prediction on the frame
#         prediction, confidence = run_model_on_frame(frame)

#         # Encode frame as JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         image_base64 = base64.b64encode(buffer).decode('utf-8')

#         # Check confidence level
#         if confidence < CONFIDENCE_THRESHOLD:
#             return {"prediction": "No Prediction", "confidence": confidence, "image": image_base64}
        
#         return {"prediction": prediction, "confidence": confidence, "image": image_base64}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")@app.get("/predict", response_model=PredictionResponse)

def predict():
    try:
        # Initialize and configure picamera2
        picam2 = Picamera2()
        picam2.start_preview()
        picam2.configure(picam2.create_still_configuration())
        picam2.start()

        # Capture the frame
        frame = picam2.capture_array()

        # Stop picamera2 after capturing
        picam2.stop()

        # Run prediction on the frame
        prediction, confidence = run_model_on_frame(frame)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Check confidence level
        if confidence < CONFIDENCE_THRESHOLD:
            return {"prediction": "No Prediction", "confidence": confidence, "image": image_base64}
        
        return {"prediction": prediction, "confidence": confidence, "image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
@app.post('/recieved_data')
async def receive_data(sensor_data: SensorData, db: Session = Depends(get_db)):
    if not app.state.data_saving_enabled:
        raise HTTPException(status_code=403, detail="Data saving is disabled.")
    try:
        sensor_record = SensorDataModel(**sensor_data.dict())
        db.add(sensor_record)
        db.commit()
        db.refresh(sensor_record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Data received successfully!",
        "received_data": sensor_data.dict()
    }

@app.get("/history", response_class=HTMLResponse)
async def get_history(request: Request, db: Session = Depends(get_db)):
    records = db.query(SensorDataModel).order_by(SensorDataModel.timestamp.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "records": records})

@app.get('/api/sensor_data')
def get_sensor_data(db: Session = Depends(get_db)):
    records = db.query(SensorDataModel).all()
    data = [{
        'id': record.id,
        'serial_number': record.serial_number,
        'temperature': record.temperature,
        'humidity': record.humidity,
        'nitrogen': record.nitrogen,
        'potassium': record.potassium,
        'moisture': record.moisture,
        'eclec': record.eclec,
        'phosphorus': record.phosphorus,
        'soilPH': record.soilPH,
        'timestamp': record.timestamp.isoformat()
    } for record in records]
    return JSONResponse(content=data)
