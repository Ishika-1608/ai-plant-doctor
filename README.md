🌿 AI Plant Doctor
A Full-Stack Machine Learning Web Application that diagnoses plant diseases from leaf images using Deep Learning. It features a modern, interactive UI with real-time analysis and treatment recommendations.

🌟 Live Demo
Check out the live application deployed on Render: https://ai-plant-doctor-jvot.onrender.com

✨ Features
🤖 AI-Powered Diagnosis: Uses a Transfer Learning model (MobileNetV2) trained to identify 38 different plant diseases.
⚡ Real-Time Analysis: Instant feedback with confidence scores via FastAPI backend.
🎨 Interactive UI: Drag-and-drop interface with glassmorphism design and scanning animations.
📋 Expert Advice: Provides specific treatment suggestions for detected diseases.
📱 Responsive Design: Works seamlessly on desktop and mobile devices.

🛠️ Tech Stack
Frontend: HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API)
Backend: Python, FastAPI, Uvicorn
Machine Learning: TensorFlow, Keras, MobileNetV2
Deployment: Render (Web Service)

🚀 How to Run Locally
Prerequisites
  Python 3.9 or higher installed.
  Git installed.

Installation Steps
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-plant-doctor.gitcd ai-plant-doctor

2. Create a virtual environment
python -m venv venv
 Windowsvenv\Scripts\activate
 Mac/Linuxsource venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

5. Run the Server
uvicorn api:app --reload

6. Open in Browser, Navigate to http://127.0.0.1:8000 in your browser.

📸 Project Structure
AI_Plant_Doctor/
├── api.py                     # Backend FastAPI logic & Model Prediction
├── best_model.keras           # Trained ML Model weights
├── class_names.json           # List of disease labels
├── index.html                 # Frontend UI
├── requirements.txt           # Python dependencies
└── README.md                  # This file

📊 Dataset
This project was trained using the New Plant Diseases Dataset from Kaggle. It contains thousands of images of healthy and diseased plant leaves across various crops like Apple, Tomato, Corn, and Grape.

🌱 Future Scope
   Integrate camera access for direct mobile photo capture.
   Support for more plant species and diseases.
   User history and garden tracking dashboard.
   Community feature to share diagnoses.

📝 License
This project is open source and available under the MIT License.

👨‍💻 Author
Built with ❤️ by Ishika Malav

Note: This project was developed as part of a Machine Learning learning journey.


