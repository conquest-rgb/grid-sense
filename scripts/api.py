# app/api.py
"""
FastAPI REST API for Power Outage Risk Prediction
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import joblib
import yaml
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Kenya Power Outage Risk API",
    description="48-hour grid risk forecasting system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (simple version)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

VALID_API_KEYS = {
    "demo_key_12345": "demo_user",
    "production_key_67890": "production_user"
}

async def get_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Validate API key"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return VALID_API_KEYS[api_key]

# Load model at startup
class ModelState:
    model = None
    config = None
    feature_names = None
    threshold = None

@app.on_event("startup")
async def load_model():
    """Load model artifacts on startup"""
    logger.info("Loading model artifacts...")
    
    with open('config/config.yaml', 'r') as f:
        ModelState.config = yaml.safe_load(f)
    
    model_version = ModelState.config['model']['version']
    models_dir = ModelState.config['model']['models_dir