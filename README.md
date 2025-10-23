# grid-sense
Power outage forecasting for Kenya
# GridSense: Short-Horizon Power Outage Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Early warning system for power outages in Kenya, helping businesses and utilities prepare for disruptions 6-24 hours in advance.

---

##  Project Overview

**GridSense** predicts power outages and identifies maintenance risks using:
-  Weather data (rainfall, wind, storms)
-  Grid topology from OpenStreetMap
-  Satellite night-lights for validation
-  Scraped planned outage notices

### Why This Matters
-  **Businesses**: Reduce diesel costs, prevent spoilage, optimize operations
-  **Critical Infrastructure**: Hospitals, data centers, telecom towers can prepare
-  **Utilities**: Target maintenance and vegetation clearing

---

##  Features

- **6-24 Hour Outage Predictions** at county/substation level
- **Calibrated Probabilities** (not just yes/no)
- **Planned Outage Dashboard** with clean locations and times
- **Maintenance Risk Signals** for utility operators
- **Interactive Map** showing risk levels by region

---

##  Project Structure

```
grid-sense/
├── config/              # Configuration files
├── data/                # Data directory (not tracked in git)
│   ├── raw/            # Scraped & downloaded data
│   ├── processed/      # Cleaned & engineered features
│   ├── external/       # Third-party datasets
│   └── labels/         # Validation labels
├── dashboard/          # Streamlit web application
│   ├── app.py         # Main dashboard
│   └── pages/         # Multi-page components
├── docs/               # Documentation
├── models/             # Trained models (not tracked)
├── notebooks/          # Jupyter notebooks for analysis
├── outputs/            # Generated figures and reports
├── scripts/            # Utility scripts
│   ├── scrape_outages.py
│   ├── download_weather.py
│   └── train_model.py
├── src/                # Core source code
│   ├── data/          # Data processing modules
│   ├── models/        # ML models
│   ├── features/      # Feature engineering
│   ├── evaluation/    # Metrics & validation
│   └── utils/         # Helper functions
└── tests/              # Unit tests
```

---

##  Getting Started

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended) or **venv**
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/conquest-rgb/grid-sense.git
cd grid-sense
```

### Step 2: Create Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda create -n gridsense python=3.10 -y

# Activate environment
conda activate gridsense

# Install geospatial dependencies
conda install -c conda-forge geopandas shapely rasterio xarray folium -y
```

#### Option B: Using venv

```bash
# Create environment
python -m venv gridsense-env

# Activate environment
# On Mac/Linux:
source gridsense-env/bin/activate
# On Windows:
gridsense-env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For exact versions (recommended for team consistency)
pip install -r requirements-lock.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, geopandas, lightgbm, streamlit; print('✅ All packages installed successfully!')"
```

### Step 5: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (API keys, paths, etc.)
nano .env  # or use your preferred editor
```

### Step 6: Set Up Data Directories

```bash
# Data directories are already created with .gitkeep files
# Verify structure:
ls -la data/
# You should see: raw/, processed/, external/, labels/
```

---

##  Usage

### Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in this order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_baseline_model.ipynb
```

### Scrape Outage Notices

```bash
python scripts/scrape_outages.py
```

### Download Weather Data

```bash
python scripts/download_weather.py
```

### Train Models

```bash
python scripts/train_model.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```



## Tech Stack

### Core Libraries
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, LightGBM, XGBoost, SHAP
- **Geospatial**: GeoPandas, Shapely, OSMnx, Folium, Rasterio
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Dashboard**: Streamlit
- **Web Scraping**: BeautifulSoup4, requests

### Development Tools
- **Version Control**: Git, GitHub
- **Notebooks**: Jupyter, IPython
- **Testing**: pytest (optional)
- **Environment**: Conda/venv

---

##  Model Performance

### Metrics
- **PR-AUC**: Precision-Recall Area Under Curve (handles class imbalance)
- **Recall @ 5% Alert Rate**: Catch many events with few false alarms
- **Brier Score**: Calibration quality
- **MAE/RMSE**: For duration predictions

### Baseline
Simple heuristic using rainfall thresholds + day-of-week patterns

### Advanced Models
- LightGBM/XGBoost classifiers with calibrated probabilities
- Unsupervised clustering for site vulnerability personas

---

##  Development Timeline

### Week 1: MVP
-  Set up project structure
- Web scraper for planned outages
-  Weather feature engineering
-  Baseline county-level risk model
-  Simple Streamlit dashboard

### Week 2: Enhancement
- Probability calibration
- Night-lights validation
- Planned outage module
- CSV/JSON export functionality

### Week 3: Polish
- Substation-level granularity
- Site vulnerability personas
- Final dashboard polish
- Documentation & deployment

---

##  Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes and commit
git add .
git commit -m "Description of changes"

# 3. Push to GitHub
git push origin feature/your-feature-name

# 4. Create Pull Request on GitHub
```

---

## 👥 Team

**AI Avengers** - Moringa School Data Science Bootcamp

- [Team Member 1]
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **KPLC**: Public outage notices
- **OpenStreetMap**: Grid infrastructure data
- **CHIRPS/ERA5**: Weather data
- **VIIRS**: Night-lights validation data
- **Moringa School**: Project guidance and support




##  Future Enhancements

- [ ] Real-time API for predictions
- [ ] WhatsApp/SMS alert system
- [ ] Mobile app for field technicians
- [ ] Integration with utility SCADA systems
- [ ] Historical outage database
- [ ] Community reporting features

---

** GridSense - Powering Reliability Through Prediction **
