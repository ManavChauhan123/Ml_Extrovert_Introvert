# backend/app/schemas/classification.py
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class ClassificationInput(BaseModel):
    features: Dict[str, float]

class ClassificationOutput(BaseModel):
    predicted_class: str
    prediction_probabilities: Dict[str, float]
    confidence: float
    model_used: str 

class EDAResponse(BaseModel):
    correlation_matrix: Dict[str, Dict[str, float]]
    vif_analysis: List[Dict[str, Any]]
    basic_info: Dict[str, Any]
    target_distribution: Dict[str, int]

class ModelMetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    best_params: Dict[str, Any]
    confusion_matrix: List[List[int]]

class ModelComparisonResponse(BaseModel):
    models: List[ModelMetricsResponse]
    best_model: str
    best_metric: str