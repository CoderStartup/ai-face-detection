# Face Recognition API

<img width="1176" alt="Screenshot 2025-04-26 at 1 55 20 AM" src="https://github.com/user-attachments/assets/e656513e-1b7b-4027-b3c6-bedf2821b07a" />

A FastAPI-based application that provides face recognition capabilities, including training face models, detecting faces in images, and real-time face recognition through a webcam.

## Features

- **Face Registration**: Upload face images with names to train the recognition model
- **Face Detection**: Detect and identify faces in uploaded images
- **Real-time Recognition**: Detect and identify faces in real-time using a webcam

- **Model Management**: View and delete trained face models

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-recognition-api.git
   cd face-recognition-api
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install fastapi uvicorn face_recognition numpy pillow python-multipart jinja2
   ```

   Note: The `face_recognition` library requires `dlib` which may need additional system dependencies. On some systems, you might need to install cmake and other build tools.

### Running the Application

1. Start the server:
   ```bash
   python facedetection/app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## API Endpoints

### Web Interface
- `GET /`: Main web interface for interacting with the application

### API Endpoints
- `POST /upload-training-face/`: Register a new face
- `POST /detect-face/`: Detect faces in an uploaded image
- `GET /known-faces/`: Get a list of all registered face names
- `DELETE /known-faces/{name}`: Delete a registered face by name
- `WebSocket /ws/detect-face-realtime/`: Real-time face detection from webcam

## Project Structure

```
facedetection/
├── app.py              # Main application file
├── models/             # Saved face recognition models
├── static/             # Static files (CSS, JS, uploads)
│   ├── known_faces/    # Stored face images for training
│   └── uploads/        # Temporary uploaded images
└── templates/          # HTML templates
    └── index.html      # Main web interface
```

## Technical Details

- The application uses the `face_recognition` library, which is built on top of dlib
- Face encodings are generated from images and stored in a pickle file
- Face matching uses a tolerance of 0.6 (lower values are more strict)
- Confidence scores are calculated as (1 - face_distance)

## Troubleshooting

- If you encounter issues with the `face_recognition` library, ensure you have the required system dependencies for dlib
- For webcam access issues, ensure your browser has permission to access your camera
- The application creates necessary directories on startup, but if you encounter permission issues, create them manually

If you're using an Apple M1/M2 MacBook or other Apple silicon device, you'll need to take some additional steps for the face_recognition library to work properly:

1. Install Homebrew if you don't have it already:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install CMake and other dependencies:
   ```bash
   brew install cmake
   brew install openblas
   brew install python
   ```

3. You may also need to set some environment variables:
   ```bash
   export OPENBLAS=$(brew --prefix openblas)
   export CFLAGS="-I$OPENBLAS/include"
   export LDFLAGS="-L$OPENBLAS/lib"
   ```

4. Install dlib separately (with specific options for Apple silicon):
   ```bash
   pip install dlib
   ```

5. Then install the remaining dependencies:
   ```bash
   pip install fastapi uvicorn face_recognition numpy pillow python-multipart jinja2
   ```

If you encounter any issues, you might need to install dlib from source with specific compilation options for Apple silicon. Reference the [dlib documentation](https://github.com/davisking/dlib) for more details.

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2023 Face Recognition API Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The MIT License is a permissive open-source license that allows for reuse with minimal restrictions. It permits users to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.

## Acknowledgments

- Face recognition powered by [face_recognition](https://github.com/ageitgey/face_recognition)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
