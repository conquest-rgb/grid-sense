import importlib
import sys

print("\n Checking required packages...\n")

# mapping between package names (pip) and their import names
packages = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "joblib": "joblib",
    "pathlib": "pathlib",
    "scikit-learn": "sklearn",
    "qrcode": "qrcode",
    "Pillow": "PIL",
}

for pkg_name, import_name in packages.items():
    try:
        module = importlib.import_module(import_name)
        location = getattr(module, "__file__", "built-in")
        print(f"{pkg_name} is installed. [Location: {location}]")
    except ImportError:
        print(f"{pkg_name} is NOT installed. Please run: pip install {pkg_name}")

print("\n Check complete! If all packages show as installed, you're ready to deploy.\n")
print(f"Using Python from: {sys.executable}")
