from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pickle
import pandas as pd
from typing import Dict
from .core.database import get_db,Base, engine
from .schemas.classification import ClassificationInput, ClassificationOutput
from .models.database import ClassificationHistory
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ML Classification API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory and templates
frontend_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "static")), name="static")
templates = Jinja2Templates(directory=frontend_path)

# Load the trained classification model
with open('models/classification_model.pickle', 'rb') as f:
    model_data = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify", response_model=ClassificationOutput)
async def classify(input_data: ClassificationInput, db: Session = Depends(get_db)):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # Scale the input
        scaled_input = model_data['scaler'].transform(df)
        
        # Make prediction
        prediction = model_data['model'].predict(scaled_input)[0]
        prediction_proba = model_data['model'].predict_proba(scaled_input)[0]
        
        # Convert prediction back to original label if label encoder was used
        if model_data['label_encoder']:
            predicted_class = model_data['label_encoder'].inverse_transform([prediction])[0]
            class_names = model_data['classes']
        else:
            predicted_class = str(prediction)
            class_names = [f"Class_{i}" for i in range(len(prediction_proba))]
        
        # Create probability dictionary
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(prediction_proba)}
        
        # Calculate confidence (max probability)
        confidence = float(max(prediction_proba))
        
        # Save to database
        db_classification = ClassificationHistory(
            input_data=input_data.features,
            predicted_class=predicted_class,
            prediction_probability=prob_dict,
            confidence=confidence,
            model_used=model_data['model_name']
        )
        db.add(db_classification)
        db.commit()
        
        return ClassificationOutput(
            predicted_class=predicted_class,
            prediction_probabilities=prob_dict,
            confidence=confidence,
            model_used=model_data['model_name']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    return {
        "model_name": model_data['model_name'],
        "best_params": model_data['best_params'],
        "feature_names": model_data['feature_names'],
        "classes": model_data['classes'],
        "results": model_data['results']
    }

@app.get("/model-comparison")
async def get_model_comparison():
    """Get comparison of all trained models"""
    models_comparison = []
    
    for name, results in model_data['results'].items():
        model_metrics = {
            "model_name": name,
            "accuracy": results['test_accuracy'],
            "precision": results['test_precision'],
            "recall": results['test_recall'],
            "f1_score": results['test_f1'],
            "roc_auc": results['test_roc_auc'],
            "best_params": results['best_params'],
            "confusion_matrix": results['confusion_matrix']
        }
        models_comparison.append(model_metrics)
    
    # Find best model based on F1 score
    best_model = max(models_comparison, key=lambda x: x['f1_score'])
    
    return {
        "models": models_comparison,
        "best_model": best_model['model_name'],
        "best_metric": "f1_score"
    }

@app.get("/classification-history")
async def get_classification_history(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent classification history"""
    history = db.query(ClassificationHistory).order_by(
        ClassificationHistory.timestamp.desc()
    ).limit(limit).all()
    
    return [
        {
            "id": record.id,
            "predicted_class": record.predicted_class,
            "confidence": record.confidence,
            "model_used": record.model_used,
            "timestamp": record.timestamp
        }
        for record in history
    ]

@app.get("/class-distribution")
async def get_class_distribution(db: Session = Depends(get_db)):
    """Get distribution of predicted classes"""
    from sqlalchemy import func
    
    distribution = db.query(
        ClassificationHistory.predicted_class,
        func.count(ClassificationHistory.predicted_class).label('count')
    ).group_by(ClassificationHistory.predicted_class).all()
    
    return {
        "distribution": [
            {"class": record.predicted_class, "count": record.count}
            for record in distribution
        ]
    }

@app.get("/eda")
async def get_eda_results():
    return {
        "message": "EDA results for classification",
        "charts_available": [
            "target_distribution", 
            "correlation_matrix", 
            "feature_target_analysis",
            "confusion_matrices",
            "roc_curves"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_data is not None}