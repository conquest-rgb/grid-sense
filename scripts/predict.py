# scripts/predict.py
"""
Command-line prediction tool
Usage: python scripts/predict.py --input data.csv --output predictions.csv
"""

import argparse
import pandas as pd
import joblib
import yaml
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    """Load model and make predictions"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_version = self.config['model']['version']
        models_dir = self.config['model']['models_dir']
        
        # Load model artifacts
        self.model = joblib.load(f'{models_dir}/rf_tuned_final_v{model_version}.pkl')
        self.threshold = self.config['model']['threshold']
        
        with open(f'{models_dir}/feature_names_v{model_version}.json', 'r') as f:
            import json
            feature_config = json.load(f)
            self.feature_names = feature_config['all_features']
        
        logger.info(f"✅ Model loaded (v{model_version}, threshold={self.threshold})")
    
    def predict(self, X):
        """Make predictions"""
        # Validate features
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X_selected = X[self.feature_names]
        
        # Predict
        probabilities = self.model.predict_proba(X_selected)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Risk levels
        risk_levels = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Results
        results = pd.DataFrame({
            'risk_probability': probabilities,
            'risk_level': risk_levels,
            'is_high_risk': predictions
        })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict power outage risk')
    parser.add_argument('--input', required=True, help='Input CSV file with features')
    parser.add_argument('--output', required=True, help='Output CSV file for predictions')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    X = pd.read_csv(args.input)
    logger.info(f"✅ Loaded {len(X)} records")
    
    # Predict
    logger.info("Making predictions...")
    predictor = Predictor(config_path=args.config)
    predictions = predictor.predict(X)
    
    # Combine with input
    output = pd.concat([X, predictions], axis=1)
    
    # Save
    output.to_csv(args.output, index=False)
    logger.info(f"✅ Predictions saved to {args.output}")
    
    # Summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total records: {len(predictions)}")
    print(f"\nRisk Distribution:")
    print(predictions['risk_level'].value_counts())
    print(f"\nHigh-Risk Alerts: {predictions['is_high_risk'].sum()} ({predictions['is_high_risk'].mean():.1%})")
    print("="*80)

if __name__ == "__main__":
    main()