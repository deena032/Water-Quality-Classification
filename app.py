"""
Water Quality Classification - Flask Web Application
==================================================
Serves the responsive blue-and-white web interface and performs real-time
water potability predictions using the trained Random Forest pipeline saved in
`model.pkl`.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

FEATURE_SPECS = [
    {
        "name": "ph",
        "label": "pH Level",
        "unit": "0 - 14",
        "min": 0.0,
        "max": 14.0,
        "placeholder": "e.g. 7.12",
        "hint": "WHO recommended drinking water range: 6.5 - 8.5",
    },
    {
        "name": "Hardness",
        "label": "Hardness",
        "unit": "mg/L",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 196.37",
        "hint": "Calcium and magnesium concentration",
    },
    {
        "name": "Solids",
        "label": "Total Dissolved Solids",
        "unit": "ppm",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 20927.83",
        "hint": "Mineral & salt content (TDS)",
    },
    {
        "name": "Chloramines",
        "label": "Chloramines",
        "unit": "ppm",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 7.12",
        "hint": "Disinfectant level (WHO limit: <= 4 ppm)",
    },
    {
        "name": "Sulfate",
        "label": "Sulfate",
        "unit": "mg/L",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 333.78",
        "hint": "Naturally occurring dissolved sulfate",
    },
    {
        "name": "Conductivity",
        "label": "Conductivity",
        "unit": "uS/cm",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 426.20",
        "hint": "Electrical conductivity of water",
    },
    {
        "name": "Organic_carbon",
        "label": "Organic Carbon",
        "unit": "ppm",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 14.28",
        "hint": "Total organic carbon (TOC)",
    },
    {
        "name": "Trihalomethanes",
        "label": "Trihalomethanes",
        "unit": "ug/L",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 66.40",
        "hint": "Disinfection byproducts (WHO limit: <= 80 ug/L)",
    },
    {
        "name": "Turbidity",
        "label": "Turbidity",
        "unit": "NTU",
        "min": 0.0,
        "max": None,
        "placeholder": "e.g. 3.97",
        "hint": "Water cloudiness / suspended solids",
    },
]

FEATURE_NAMES = [field["name"] for field in FEATURE_SPECS]

app = Flask(__name__)


def load_model():
    """Load the Joblib-saved Random Forest pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python train_model.py` first to generate model.pkl."
        )
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict[str, Any]:
    """Load saved evaluation metrics if available."""
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


model = load_model()
model_metrics = load_metrics()


def parse_and_validate_inputs(raw_data: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Parse and validate all 9 numeric water-quality inputs."""
    parsed: dict[str, float] = {}
    errors: list[str] = []

    for spec in FEATURE_SPECS:
        name = spec["name"]
        label = spec["label"]
        raw_val = str(raw_data.get(name, "")).strip()

        if raw_val == "":
            errors.append(f"{label} is required.")
            continue

        try:
            value = float(raw_val)
        except ValueError:
            errors.append(f"{label} must be a valid number.")
            continue

        if pd.isna(value) or value in (float("inf"), float("-inf")):
            errors.append(f"{label} must be a finite number.")
            continue

        if spec["min"] is not None and value < spec["min"]:
            errors.append(f"{label} cannot be less than {spec['min']}.")
            continue

        if spec["max"] is not None and value > spec["max"]:
            errors.append(f"{label} cannot be greater than {spec['max']}.")
            continue

        parsed[name] = value

    return parsed, errors


def run_prediction(features: dict[str, float]) -> dict[str, Any]:
    """Run real inference through the loaded Random Forest pipeline."""
    input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    predicted_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    not_potable_prob = round(float(probabilities[0]) * 100, 1)
    potable_prob = round(float(probabilities[1]) * 100, 1)
    is_potable = predicted_class == 1

    return {
        "is_potable": is_potable,
        "label": "Potable" if is_potable else "Not Potable",
        "headline": (
            "Potable — Safe for Consumption"
            if is_potable
            else "Not Potable — Unsafe for Consumption"
        ),
        "confidence": potable_prob if is_potable else not_potable_prob,
        "potable_prob": potable_prob,
        "not_potable_prob": not_potable_prob,
        "predicted_class": predicted_class,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the home page and handle form predictions."""
    form_values: dict[str, str] = {}
    prediction: dict[str, Any] | None = None
    errors: list[str] = []

    if request.method == "POST":
        form_values = {
            spec["name"]: request.form.get(spec["name"], "").strip()
            for spec in FEATURE_SPECS
        }
        parsed, errors = parse_and_validate_inputs(form_values)
        if not errors:
            prediction = run_prediction(parsed)

    return render_template(
        "index.html",
        feature_specs=FEATURE_SPECS,
        form_values=form_values,
        prediction=prediction,
        errors=errors,
        metrics=model_metrics,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint for programmatic access."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    parsed, errors = parse_and_validate_inputs(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    result = run_prediction(parsed)
    return jsonify({"inputs": parsed, "prediction": result}), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    app.run(debug=True)
