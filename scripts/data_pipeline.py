# scripts/data_pipeline.py
"""
Data Pipeline for Power Outage Risk Prediction
Fetches weather, outages, and other data sources hourly
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
import yaml
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Automated data collection and processing pipeline"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.counties = self.config['counties']
        self.weather_api_url = self.config['data_pipeline']['weather_api_url']
        
        logger.info("Data Pipeline initialized")
    
    def fetch_weather_forecast(self, hours_ahead=48):
        """
        Fetch weather forecast for all counties
        
        Parameters:
        -----------
        hours_ahead : int
            Number of hours to forecast ahead
            
        Returns:
        --------
        pd.DataFrame with weather data for all counties
        """
        logger.info(f"Fetching {hours_ahead}h weather forecast for {len(self.counties)} counties...")
        
        weather_data = []
        
        # County coordinates (sample - you'll need actual coordinates)
        county_coords = self._get_county_coordinates()
        
        for county in self.counties:
            try:
                lat, lon = county_coords.get(county, (0, 36))  # Default to Kenya center
                
                # Call OpenMeteo API
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'hourly': 'temperature_2m,precipitation,windspeed_10m,windgusts_10m,cloudcover,cape',
                    'forecast_days': 2,
                    'timezone': 'Africa/Nairobi'
                }
                
                response = requests.get(self.weather_api_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Parse response
                hourly = data['hourly']
                for i in range(min(hours_ahead, len(hourly['time']))):
                    weather_data.append({
                        'county': county,
                        'timestamp': hourly['time'][i],
                        'temperature': hourly['temperature_2m'][i],
                        'precipitation': hourly['precipitation'][i],
                        'wind_speed': hourly['windspeed_10m'][i],
                        'wind_gusts': hourly['windgusts_10m'][i],
                        'cloud_cover': hourly['cloudcover'][i],
                        'cape': hourly.get('cape', [0]*len(hourly['time']))[i]
                    })
                
                logger.info(f"  ✓ {county}")
                
            except Exception as e:
                logger.error(f"  ✗ {county}: {str(e)}")
                continue
        
        df = pd.DataFrame(weather_data)
        logger.info(f"Weather data fetched: {len(df)} records")
        
        return df
    
    def fetch_planned_outages(self):
        """
        Scrape KPLC website for current planned outages
        
        Returns:
        --------
        pd.DataFrame with planned outages
        """
        logger.info("Fetching planned outages from KPLC...")
        
        # This is a placeholder - actual implementation would scrape KPLC website
        # For now, return empty dataframe or use your existing data
        
        try:
            # In production, you'd use BeautifulSoup or Selenium here
            # For demo, return sample data
            outages = []
            
            # Example: You could load from your existing CSV
            # outages_df = pd.read_csv('data/raw/kplc_planned_outages.csv')
            
            df = pd.DataFrame(outages)
            logger.info(f"Planned outages fetched: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch planned outages: {str(e)}")
            return pd.DataFrame()
    
    def fetch_ndvi_data(self):
        """
        Fetch NDVI vegetation data
        
        Returns:
        --------
        pd.DataFrame with NDVI data by county
        """
        logger.info("Fetching NDVI data...")
        
        # Placeholder - in production, fetch from satellite API or use existing data
        try:
            # Load from your existing data or API
            # ndvi_df = pd.read_csv('data/raw/ndvi_data.csv')
            
            # For demo, use default values by county
            ndvi_data = []
            for county in self.counties:
                ndvi_data.append({
                    'county': county,
                    'ndvi_mean': 0.45,  # Placeholder
                    'ndvi_std': 0.12,
                    'ndvi_min': 0.2,
                    'ndvi_max': 0.7,
                    'timestamp': datetime.now()
                })
            
            df = pd.DataFrame(ndvi_data)
            logger.info(f"NDVI data fetched: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch NDVI data: {str(e)}")
            return pd.DataFrame()
    
    def _get_county_coordinates(self):
        """Get approximate coordinates for each county (center point)"""
        # This is simplified - you should have actual county centroids
        coords = {
            'Nairobi': (-1.2921, 36.8219),
            'Mombasa': (-4.0435, 39.6682),
            'Kisumu': (-0.0917, 34.7680),
            'Nakuru': (-0.3031, 36.0800),
            'Eldoret': (0.5143, 35.2698),
            # Add all 47 counties...
            # For demo, default to Kenya center for others
        }
        return coords
    
    def aggregate_weather_features(self, weather_df):
        """
        Aggregate weather data into 6h, 12h, 24h windows
        
        Parameters:
        -----------
        weather_df : pd.DataFrame
            Hourly weather data
            
        Returns:
        --------
        pd.DataFrame with aggregated features
        """
        logger.info("Aggregating weather features...")
        
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        weather_df = weather_df.sort_values(['county', 'timestamp'])
        
        aggregated = []
        
        for county in weather_df['county'].unique():
            county_data = weather_df[weather_df['county'] == county].copy()
            
            for idx, row in county_data.iterrows():
                current_time = row['timestamp']
                
                # Get windows
                data_6h = county_data[
                    (county_data['timestamp'] <= current_time) &
                    (county_data['timestamp'] > current_time - timedelta(hours=6))
                ]
                data_12h = county_data[
                    (county_data['timestamp'] <= current_time) &
                    (county_data['timestamp'] > current_time - timedelta(hours=12))
                ]
                data_24h = county_data[
                    (county_data['timestamp'] <= current_time) &
                    (county_data['timestamp'] > current_time - timedelta(hours=24))
                ]
                
                # Aggregate
                features = {
                    'county': county,
                    'timestamp': current_time,
                    'hour_of_day': current_time.hour,
                    'month': current_time.month,
                    'weekday': current_time.weekday(),
                    'is_weekend': int(current_time.weekday() >= 5),
                    
                    # CAPE
                    'cape_mean_6h_mean': data_6h['cape'].mean() if len(data_6h) > 0 else 0,
                    'cape_mean_12h_mean': data_12h['cape'].mean() if len(data_12h) > 0 else 0,
                    'cape_mean_24h_mean': data_24h['cape'].mean() if len(data_24h) > 0 else 0,
                    
                    # Wind
                    'wind_mean_6h_mean': data_6h['wind_speed'].mean() if len(data_6h) > 0 else 0,
                    'wind_mean_12h_mean': data_12h['wind_speed'].mean() if len(data_12h) > 0 else 0,
                    'wind_mean_24h_mean': data_24h['wind_speed'].mean() if len(data_24h) > 0 else 0,
                    
                    'wind_max_6h_max': data_6h['wind_gusts'].max() if len(data_6h) > 0 else 0,
                    'wind_max_12h_max': data_12h['wind_gusts'].max() if len(data_12h) > 0 else 0,
                    'wind_max_24h_max': data_24h['wind_gusts'].max() if len(data_24h) > 0 else 0,
                    
                    # Cloud
                    'cloud_mean_6h_mean': data_6h['cloud_cover'].mean() if len(data_6h) > 0 else 0,
                    'cloud_mean_12h_mean': data_12h['cloud_cover'].mean() if len(data_12h) > 0 else 0,
                    'cloud_mean_24h_mean': data_24h['cloud_cover'].mean() if len(data_24h) > 0 else 0,
                    
                    # Precipitation
                    'total_precipitation_mm_mean': data_24h['precipitation'].mean() if len(data_24h) > 0 else 0,
                    'total_precipitation_mm_max': data_24h['precipitation'].max() if len(data_24h) > 0 else 0,
                }
                
                aggregated.append(features)
        
        df = pd.DataFrame(aggregated)
        logger.info(f"Features aggregated: {len(df)} records")
        
        return df
    
    def create_feature_matrix(self):
        """
        Create complete feature matrix ready for prediction
        
        Returns:
        --------
        pd.DataFrame with all features required by model
        """
        logger.info("="*80)
        logger.info("CREATING FEATURE MATRIX")
        logger.info("="*80)
        
        # Fetch all data sources
        weather_df = self.fetch_weather_forecast(hours_ahead=48)
        outages_df = self.fetch_planned_outages()
        ndvi_df = self.fetch_ndvi_data()
        
        # Aggregate weather
        features_df = self.aggregate_weather_features(weather_df)
        
        # Merge NDVI
        if not ndvi_df.empty:
            features_df = features_df.merge(
                ndvi_df[['county', 'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max']],
                on='county',
                how='left'
            )
        
        # Add static features (from your training data)
        # In production, load from database or reference file
        static_features = self._load_static_features()
        features_df = features_df.merge(static_features, on='county', how='left')
        
        # Create lag features (placeholder - in production, query from database)
        features_df = self._add_lag_features(features_df)
        
        # Create derived features
        features_df['storm_index'] = (
            features_df['wind_max_24h_max'] * 
            features_df['total_precipitation_mm_mean'] * 
            features_df['cape_mean_24h_mean']
        )
        
        # Cyclical encoding
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"✅ Feature matrix created: {features_df.shape}")
        logger.info("="*80)
        
        return features_df
    
    def _load_static_features(self):
        """Load static features (population, grid infrastructure, etc.)"""
        # In production, load from database or reference file
        # For demo, return your training data statistics
        
        # Placeholder - load from your actual data
        static = pd.DataFrame({
            'county': self.counties,
            'population': [100000] * len(self.counties),  # Placeholder
            'area_km2': [1000] * len(self.counties),
            'density_per_km2': [100] * len(self.counties),
            'transformer_count_per_county': [50] * len(self.counties),
            'substation_count_per_county': [5] * len(self.counties),
            'transformers_per_100k': [50] * len(self.counties),
            'substations_per_100k': [5] * len(self.counties),
            'season_first': ['dry_cool'] * len(self.counties),
            'area_type': ['urban'] * len(self.counties),
            'is_holiday_first': [0] * len(self.counties),
        })
        
        return static
    
    def _add_lag_features(self, df):
        """Add lag features (outage history)"""
        # In production, query from database
        # For demo, initialize with zeros
        
        df['outage_lag_1h'] = 0
        df['outage_lag_6h'] = 0
        df['outage_lag_24h'] = 0
        df['outage_last_24h'] = 0
        df['outage_last_7d'] = 0
        df['wind_max_lag_24h'] = df['wind_max_24h_max']  # Simplified
        df['cape_lag_24h'] = df['cape_mean_24h_mean']    # Simplified
        df['ndvi_mean_lag_24h'] = df.get('ndvi_mean', 0)
        df['precip_rolling_sum_24h'] = df['total_precipitation_mm_mean']
        df['wind_gust_rolling_max_6h'] = df['wind_max_6h_max']
        
        return df
    
    def run(self):
        """Run complete pipeline"""
        try:
            features = self.create_feature_matrix()
            
            # Save to file
            output_path = f'data/processed/features_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
            features.to_csv(output_path, index=False)
            logger.info(f"✅ Features saved to {output_path}")
            
            return features
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

# CLI interface
if __name__ == "__main__":
    pipeline = DataPipeline()
    features = pipeline.run()
    print(f"\n✅ Pipeline complete. Features shape: {features.shape}")