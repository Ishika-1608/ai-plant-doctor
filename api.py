from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse # <--- ENSURE HTMLResponse IS HERE
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

# --- APP INITIALIZATION ---
app = FastAPI()

# --- CONFIGURATION FOR CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
print("Loading model... this may take a moment.")
model = tf.keras.models.load_model('best_model.keras')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print("Model loaded. Server ready.")

# --- TREATMENT DATABASE ---
treatments = {
    "Apple___Apple_scab": "Apply fungicides containing Captan or Mancozeb. Remove fallen leaves to prevent reinfection next season.",
    "Apple___Black_rot": "Prune out all infected wood and cankers. Apply fungicides (e.g., Myclobutanil) during bloom and petal fall.",
    "Apple___Cedar_apple_rust": "Remove nearby Red Cedar trees if possible. Apply protective fungicides (e.g., sulfur) before the buds open.",
    "Apple___healthy": "Great job! No treatment needed. Continue regular watering and monitoring.",
    
    "Blueberry___healthy": "Your blueberry plant is healthy! Ensure soil pH stays acidic (4.5-5.5).",
    
    "Cherry___Powdery_mildew": "Apply fungicides like Trifloxystrobin or sulfur. Improve air circulation around the tree by pruning.",
    "Cherry___healthy": "Healthy cherry tree. Watch out for aphids during spring growth.",
    
    "Corn___Common_rust": "Plant resistant varieties. Apply foliar fungicides like Azoxystrobin if infection is severe.",
    "Corn___Northern_Leaf_Blight": "Rotate crops to break the disease cycle. Use hybrid varieties with high resistance.",
    "Corn___healthy": "Corn is healthy. Ensure adequate nitrogen levels for strong stalks.",
    
    "Grape___Black_rot": "Apply fungicides (Mancozeb, Myclobutanil) starting early in the season. Remove mummified berries.",
    "Grape___Esca_(Black_Measles)": "There is no chemical cure. Prune out infected wood in summer and avoid over-pruning in winter.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides like Copper or Mancozeb. Manage leaf wetness by improving drainage.",
    "Grape___healthy": "Healthy vineyard. Monitor canopy management to prevent humidity buildup.",
    
    "Orange___Haunglongbing_(Citrus_greening)": "CRITICAL. There is no cure. Remove infected trees to prevent spread to healthy trees. Control the Asian Citrus Psyllid insect.",
    
    "Peach___Bacterial_spot": "Use copper-based bactericides during dormant season. Plant resistant varieties.",
    "Peach___healthy": "Peach is healthy. Thin fruits to ensure remaining peaches grow large and healthy.",
    
    "Pepper,_bell___Bacterial_spot": "Use certified disease-free seeds. Apply copper sprays regularly. Avoid working in fields when wet.",
    "Pepper,_bell___healthy": "Healthy pepper plant. Ensure consistent watering to prevent blossom end rot.",
    
    "Potato___Early_blight": "Remove infected lower leaves. Apply fungicides like Chlorothalonil every 7-10 days.",
    "Potato___Late_blight": "DANGEROUS. Destroy infected plants immediately. Apply metalaxyl or copper fungicides to prevent spread.",
    "Potato___healthy": "Healthy potatoes. Hill up soil around stems to prevent sun exposure.",
    
    "Raspberry___healthy": "Healthy raspberry. Prune out old canes to encourage new growth.",
    
    "Soybean___healthy": "Healthy soybean crop. Monitor for aphids and spider mites.",
    
    "Squash___Powdery_mildew": "Apply neem oil or sulfur-based fungicides. Plant resistant varieties in the future.",
    
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply captan or myclobutanil fungicides at first signs.",
    "Strawberry___healthy": "Healthy strawberries. Mulch under plants to keep fruit off the soil.",
    
    "Tomato___Bacterial_spot": "Use copper sprays. Remove plant debris from the garden. Rotate crops next year.",
    "Tomato___Early_blight": "Remove lower infected leaves. Apply fungicides (Chlorothalonil or Copper) regularly.",
    "Tomato___Late_blight": "Severe infection. Remove and destroy plants. Apply fungicides (Mancozeb) if caught early.",
    "Tomato___Leaf_Mold": "Improve ventilation in the greenhouse. Apply fungicides targeting molds.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Mulch the soil to prevent spores from splashing onto leaves.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Wash leaves with water to remove mites. Use insecticidal soaps or neem oil.",
    "Tomato___Target_Spot": "Apply fungicides like Azoxystrobin. Ensure good air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Viral disease. There is no cure. Remove the plant. Control whiteflies to prevent spread.",
    "Tomato___Tomato_mosaic_virus": "Viral disease. No chemical cure. Remove infected plants and wash hands/tools to prevent spread.",
    "Tomato___healthy": "Healthy tomato plant! Provide support (cages/stakes) as it grows."
}

# --- HELPER FUNCTION ---
def predict_disease(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((128, 128))
        
        # Prepare Array
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        
        # Score
        score = predictions[0]
        
        max_index = np.argmax(score)
        predicted_class = class_names[max_index]
        confidence = float(100 * np.max(score))
        
        # Get advice
        advice = treatments.get(predicted_class, "Consult a local agricultural expert for specific treatment.")
        
        return {"class": predicted_class, "confidence": confidence, "advice": advice}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS ---

# THIS PART IS NEW AND CRUCIAL
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_disease(image_bytes)
    return JSONResponse(content=result)