import os
import shutil
import uuid
import face_recognition
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import logging
from datetime import datetime
import base64
import io
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Check if face_recognition_models is installed properly
try:
    import face_recognition_models
    logger.info(f"face_recognition_models found at: {face_recognition_models.__file__}")
except ImportError:
    logger.error("face_recognition_models not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "git+https://github.com/ageitgey/face_recognition_models"])
    logger.info("face_recognition_models installed successfully!")

app = FastAPI(title="Face Recognition API")

# Create necessary directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/known_faces", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Path to the model file
MODEL_PATH = "models/face_recognition_model.pkl"

def load_model():
    """Load the face recognition model if it exists"""
    global known_face_encodings, known_face_names
    
    if os.path.exists(MODEL_PATH):
        logger.info("Loading existing face recognition model...")
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data.get('encodings', [])
            known_face_names = data.get('names', [])
        logger.info(f"Loaded {len(known_face_names)} faces: {known_face_names}")
    else:
        logger.info("No existing model found. Starting with empty model.")

def save_model():
    """Save the current face recognition model"""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'encodings': known_face_encodings,
            'names': known_face_names,
            'timestamp': datetime.now().isoformat()
        }, f)
    logger.info(f"Model saved with {len(known_face_names)} faces")

@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "known_faces": known_face_names}
    )

@app.post("/upload-training-face/")
async def upload_training_face(name: str = Form(...), file: UploadFile = File(...)):
    """
    Upload a face image for training
    """
    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"static/known_faces/{unique_filename}"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load the image and find face encodings
        image = face_recognition.load_image_file(file_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return JSONResponse(
                status_code=400,
                content={"error": "No face detected in the image. Please upload a clear image with a face."}
            )
        
        # Use the first face found
        face_encoding = face_encodings[0]
        
        # Add to our known faces
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        
        # Save the updated model
        save_model()
        
        return {"success": True, "name": name, "file_path": file_path}
    
    except Exception as e:
        logger.error(f"Error processing uploaded face: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process the image: {str(e)}"}
        )

@app.post("/detect-face/")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image and identify known faces
    """
    try:
        # Create a temporary file
        temp_file_path = f"static/uploads/{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load the image
        image = face_recognition.load_image_file(temp_file_path)
        
        # Find all face locations and face encodings in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Initialize results
        result = {
            "detected_faces": len(face_locations),
            "identified_faces": [],
            "image_path": temp_file_path
        }
        
        # Go through each face found in the image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            confidence = 0.0
            
            # If we found a match
            if True in matches:
                # Find the best match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]  # Convert distance to confidence
            
            # Add the face info to the results
            result["identified_faces"].append({
                "name": name,
                "confidence": float(confidence),
                "location": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process the image: {str(e)}"}
        )

@app.get("/known-faces/", response_model=List[str])
async def get_known_faces():
    """Get the list of known face names"""
    return known_face_names

@app.delete("/known-faces/{name}")
async def delete_known_face(name: str):
    """Delete a known face by name"""
    global known_face_encodings, known_face_names
    
    try:
        # Find all indices with this name
        indices = [i for i, n in enumerate(known_face_names) if n == name]
        
        if not indices:
            return JSONResponse(
                status_code=404,
                content={"error": f"No face found with name: {name}"}
            )
        
        # Remove the face encodings and names in reverse order
        for idx in sorted(indices, reverse=True):
            known_face_encodings.pop(idx)
            known_face_names.pop(idx)
        
        # Save the updated model
        save_model()
        
        return {"success": True, "message": f"Deleted {len(indices)} faces for {name}"}
    
    except Exception as e:
        logger.error(f"Error deleting face: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete face: {str(e)}"}
        )

@app.websocket("/ws/detect-face-realtime/")
async def websocket_detect_face_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time face detection from webcam
    """
    await websocket.accept()
    logger.info("WebSocket connection established for real-time face detection")
    
    try:
        while True:
            # Receive the base64 encoded image from the client
            data = await websocket.receive_text()
            
            # Skip empty frames or connection check messages
            if not data or data == "ping":
                await websocket.send_text("pong")
                continue
            
            try:
                # Decode the base64 image
                image_data = data.split(",")[1] if "," in data else data
                image_bytes = base64.b64decode(image_data)
                
                # Convert to image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert PIL Image to numpy array for face_recognition
                image_np = np.array(image)
                
                # Find all face locations and face encodings in the image
                face_locations = face_recognition.face_locations(image_np)
                face_encodings = face_recognition.face_encodings(image_np, face_locations)
                
                # Initialize results
                result = {
                    "detected_faces": len(face_locations),
                    "identified_faces": []
                }
                
                # Go through each face found in the image
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Unknown"
                    confidence = 0.0
                    
                    # If we found a match
                    if True in matches and known_face_encodings:
                        # Find the best match
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]  # Convert distance to confidence
                    
                    # Add the face info to the results
                    result["identified_faces"].append({
                        "name": name,
                        "confidence": float(confidence),
                        "location": {
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "left": left
                        }
                    })
                
                # Send the results back to the client
                await websocket.send_json(result)
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({"error": f"Failed to process the image: {str(e)}"})
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)