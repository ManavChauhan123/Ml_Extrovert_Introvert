# backend/app/models/database.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from ..core.database import Base

class ClassificationHistory(Base):
    __tablename__ = "classification_history"
    
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(JSON)
    predicted_class = Column(String)
    prediction_probability = Column(JSON)  # Store probabilities for all classes
    confidence = Column(Float)
    model_used = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
class ModelMetrics(Base):
    __tablename__ = "classification_model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    best_params = Column(JSON)
    confusion_matrix = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())