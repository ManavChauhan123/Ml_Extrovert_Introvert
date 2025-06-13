from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pickle
import pandas as pd
from typing import Dict
from .core.database import get_db, Base, engine
from .schemas.classification import ClassificationInput, ClassificationOutput
from .models.database import ClassificationHistory
import os
from .models import database as models  # 👈 import models to register them
Base.metadata.create_all(bind=engine)  # 👈 tables will now be created


app = FastAPI(title="ML Classification API", version="1.0.0")

# CORS config for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ✅ Static and Template Setup --- #
# Path to frontend root (2 levels up from backend/main.py)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

# Mount /static => frontend/static/
app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "static")), name="static")

# ✅ Mount /eda => frontend/static/eda/
app.mount("/eda", StaticFiles(directory=os.path.join(frontend_path, "static", "eda")), name="eda")

# Jinja2 template path (index.html in frontend/)
templates = Jinja2Templates(directory=frontend_path)

# --- ✅ Load ML model ---
with open('models/classification_model.pickle', 'rb') as f:
    model_data = pickle.load(f)

# --- ✅ Serve Frontend HTML ---
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- ✅ ML Prediction Endpoint ---
@app.post("/classify", response_model=ClassificationOutput)
def classify(input_data: ClassificationInput, db: Session = Depends(get_db)):
    try:
        print("🔍 Received Input:", input_data.features)
           # --- Validate binary categorical fields ---

        df = pd.DataFrame([input_data.features])
        print("🧪 DataFrame created:", df)

        scaled_input = model_data['scaler'].transform(df)
        print("📊 Scaled input:", scaled_input)

        prediction = model_data['model'].predict(scaled_input)[0]
        print("🔮 Prediction:", prediction)

        prediction_proba = model_data['model'].predict_proba(scaled_input)[0]
        print("📈 Prediction Probabilities:", prediction_proba)

        # Optional: check if label encoder exists
        if model_data['label_encoder']:
            predicted_class = model_data['label_encoder'].inverse_transform([prediction])[0]
        else:
            predicted_class = str(prediction)

        confidence = float(max(prediction_proba))
        print("✅ Predicted Class:", predicted_class)
        print("🎯 Confidence:", confidence)

        class_names = model_data['classes']
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(prediction_proba)}

        # Save to DB
        db_record = ClassificationHistory(
            input_data=input_data.features,
            predicted_class=predicted_class,
            prediction_probability=prob_dict,
            confidence=confidence,
            model_used=model_data['model_name']
        )
        db.add(db_record)
        db.commit()
        

        output = ClassificationOutput(
        predicted_class=predicted_class,
        prediction_probabilities=prob_dict,
        confidence=confidence,
        model_used=model_data['model_name'])
        print("📤 Final Response Output:", output)
        return output


    except Exception as e:
        print("❌ Exception:", e)
        raise HTTPException(status_code=500, detail=str(e))



# --- ✅ Model Metadata ---
@app.get("/model-info")
async def get_model_info():
    return {
        "model_name": model_data['model_name'],
        "best_params": model_data['best_params'],
        "feature_names": model_data['feature_names'],
        "classes": model_data['classes'],
    }

@app.get("/model-comparison")
async def get_model_comparison():
    models_comparison = []
    for name, results in model_data['results'].items():
        models_comparison.append({
            "model_name": name,
            "accuracy": results['test_accuracy'],
            "precision": results['test_precision'],
            "recall": results['test_recall'],
            "f1_score": results['test_f1'],
            "roc_auc": results['test_roc_auc'],
            "best_params": results['best_params'],
            "confusion_matrix": results['confusion_matrix']
        })

    best_model = max(models_comparison, key=lambda x: x['f1_score'])

    return {
        "models": models_comparison,
        "best_model": best_model['model_name'],
        "best_metric": "f1_score"
    }

# --- ✅ Classification History ---
@app.get("/classification-history")
async def get_classification_history(limit: int = 50, db: Session = Depends(get_db)):
    history = db.query(ClassificationHistory).order_by(ClassificationHistory.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "predicted_class": r.predicted_class,
            "confidence": r.confidence,
            "model_used": r.model_used,
            "timestamp": r.timestamp
        }
        for r in history
    ]


# --- ✅ EDA Chart Info Endpoint ---
@app.get("/eda")
async def get_eda_results():
    return {
        "message": "EDA results for classification",
        "charts_available": [
            "target_distribution",
            "correlation_matrix",
            "feature_target_analysis",
            "confusion_matrix",
            "roc_curve"
        ]
    }


