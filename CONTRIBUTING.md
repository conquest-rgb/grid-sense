# Contributing to GridSense

Thank you for your interest in contributing to GridSense! This project aims to provide short-horizon power outage forecasting and maintenance signals for Kenya, helping businesses and utilities prepare for disruptions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Data Ethics & Security](#data-ethics--security)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

Before contributing, please:

1. Read the project documentation to understand our goals
2. Check existing issues and pull requests to avoid duplication
3. Join discussions on open issues to coordinate efforts

## How to Contribute

We welcome contributions in several areas:

### 1. Data Collection & Processing
- Improve web scrapers for planned outage notices
- Enhance geocoding accuracy for Kenyan locations
- Add new weather or grid proxy data sources
- Validate night-lights data against known outages

### 2. Modeling & Analytics
- Improve probability calibration techniques
- Experiment with new algorithms (while maintaining interpretability)
- Develop site vulnerability personas using clustering
- Enhance duration prediction for planned outages

### 3. Visualization & Dashboard
- Improve the Streamlit dashboard UI/UX
- Add new visualizations (reliability diagrams, PR curves, etc.)
- Create mobile-responsive views
- Develop alert notification systems

### 4. Documentation
- Improve setup instructions
- Add tutorials or examples
- Document data sources and preprocessing steps
- Write case studies or ROI examples

### 5. Testing & Validation
- Add unit tests for data processing functions
- Create integration tests for the pipeline
- Validate predictions against historical events
- Test dashboard functionality

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/gridsense.git
   cd gridsense
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development tools
   ```

4. **Set up data versioning (if using DVC)**
   ```bash
   dvc pull  # Pull data artifacts if configured
   ```

5. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## Project Structure

```
gridsense/
├── data/
│   ├── raw/              # Scraped and downloaded data
│   ├── processed/        # Cleaned and feature-engineered data
│   └── external/         # Third-party datasets
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/            # Data collection and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   ├── visualization/   # Plotting and dashboard components
│   └── utils/           # Helper functions
├── tests/               # Unit and integration tests
├── streamlit_app/       # Streamlit dashboard code
└── docs/                # Documentation
```

## Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Code Quality
```python
# Good example
def calculate_outage_risk(
    weather_features: pd.DataFrame,
    grid_proxies: pd.DataFrame,
    model: lgb.Booster
) -> np.ndarray:
    """Calculate outage risk probabilities for each location.
    
    Args:
        weather_features: Weather data with columns [rain, wind, ...]
        grid_proxies: Grid topology features
        model: Trained LightGBM model
        
    Returns:
        Array of risk probabilities [0, 1]
    """
    features = merge_features(weather_features, grid_proxies)
    return model.predict(features)
```

### Commit Messages
Use clear, descriptive commit messages:
- `feat: add night-lights validation module`
- `fix: correct geocoding for Nairobi county`
- `docs: update API documentation`
- `test: add unit tests for weather scraper`

## Testing

All code should include appropriate tests:

```python
# tests/test_data_processing.py
def test_geocode_county():
    """Test county geocoding accuracy."""
    location = "Nairobi"
    result = geocode_kenyan_location(location)
    assert result['county'] == 'Nairobi'
    assert result['lat'] is not None
```

Run tests before submitting:
```bash
pytest tests/ --cov=src
```

## Submitting Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linting**
   ```bash
   pytest tests/
   flake8 src/
   black src/  # Auto-format code
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes
   - Ensure all CI checks pass

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include relevant test cases
- Update documentation if needed
- Respond to review comments promptly
- Ensure your branch is up-to-date with main

## Data Ethics & Security

This project involves sensitive infrastructure data. Please adhere to these principles:

### Data Privacy
- **Never commit** raw utility data or customer logs to the repository
- Aggregate data to county/substation level minimum
- Use `.gitignore` to exclude sensitive files
- Store sensitive data under NDA in secure, local folders

### Transparency
- Document all data sources clearly
- Be explicit about model uncertainties
- Use multiple validation sources to reduce noise
- Clearly communicate alert reliability

### Responsible Deployment
- Avoid alert fatigue through careful threshold tuning
- Provide clear unsubscribe mechanisms
- Do not disclose sensitive grid infrastructure details
- Consider the impact on affected communities

### Security
- Do not expose API keys or credentials
- Use environment variables for sensitive configuration
- Report security vulnerabilities privately to maintainers

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Contact maintainers for sensitive topics

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for helping make power more reliable in Kenya!
