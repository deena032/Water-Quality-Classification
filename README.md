# Water Quality Classification

A beginner-friendly, end-to-end Machine Learning web application that predicts whether a water sample is **Potable (Safe to Drink)** or **Not Potable (Unsafe to Drink)** using a **Random Forest Classifier** trained on the real Water Potability dataset.

## Features

- **Real ML Pipeline**: Uses `scikit-learn` (`SimpleImputer` + `RandomForestClassifier`) trained on `water_potability.csv` (3,276 samples, 9 physicochemical features).
- **Missing Value Handling**: Handles missing values in `ph`, `Sulfate`, and `Trihalomethanes` via median imputation fitted inside the training pipeline to prevent data leakage.
- **Model Persistence**: Saves the trained pipeline to `model.pkl` using `joblib`.
- **Responsive Blue-and-White UI**: Built with clean HTML & CSS (`templates/index.html`, `static/style.css`) featuring sample presets, validation feedback, and prediction confidence bars.
- **REST API & Web Form**: Supports interactive web predictions (`POST /`) and JSON predictions (`POST /api/predict`).

## Technologies Used

- **Language**: Python 3
- **Web Framework**: Flask (`3.1.3`)
- **Machine Learning & Data Processing**: Scikit-Learn (`1.9.1`), Pandas (`3.1.1`), NumPy (`2.5.5`)
- **Model Serialization**: Joblib (`1.6.0`)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Input Parameters

| Feature | Unit | Description |
| --- | --- | --- |
| `ph` | 0 - 14 | Acidity / alkalinity of the water |
| `Hardness` | mg/L | Calcium and magnesium mineral concentration |
| `Solids` | ppm | Total dissolved solids (TDS) |
| `Chloramines` | ppm | Concentration of chloramine disinfectants |
| `Sulfate` | mg/L | Naturally occurring dissolved sulfates |
| `Conductivity` | uS/cm | Electrical conductivity |
| `Organic_carbon` | ppm | Total organic carbon (TOC) |
| `Trihalomethanes` | ug/L | Chemical disinfection byproducts |
| `Turbidity` | NTU | Light-scattering cloudiness measure |

## Project Structure

```text
Water Quality Classification/
├── app.py                   # Flask backend and prediction routes
├── train_model.py           # Dataset loading, imputation, training & evaluation
├── model.pkl                # Joblib-serialized Random Forest pipeline
├── model_metrics.json       # Saved test accuracy & dataset summary
├── water_potability.csv     # Real Water Potability dataset (3,276 rows)
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Responsive web interface
├── static/
│   └── style.css            # Blue-and-white theme styles
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Water Quality Classification"
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

`model.pkl` is already included, or you can retrain it from scratch:

```bash
python train_model.py
```

### 4. Run the Flask web app

```bash
python app.py
```

Open your browser and visit `http://127.0.0.1:5000`.

## Model Evaluation Summary

Evaluated on a stratified 80/20 train-test split (`2,620` train / `656` test samples):

- **Test Accuracy**: `67.38%`
- **ROC-AUC Score**: `0.6648`

## API Example

Send a `POST` request to `/api/predict`:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.08,
    "Hardness": 204.89,
    "Solids": 20791.32,
    "Chloramines": 7.30,
    "Sulfate": 368.52,
    "Conductivity": 564.31,
    "Organic_carbon": 10.38,
    "Trihalomethanes": 86.99,
    "Turbidity": 2.96
  }'
```
