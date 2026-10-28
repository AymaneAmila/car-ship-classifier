from pathlib import Path
import joblib
import numpy as np
from feature_extraction import extraire_caracteristiques

MODEL_PATH = Path(__file__).parent / "artifacts" / "six_models.joblib"


def load_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if len(bundle["models"]) != 6:
        raise ValueError("Expected six trained models.")

    if bundle["feature_count"] != 59:
        raise ValueError("Expected 59 input features.")

    return bundle


def predict_with_all_models(image_rgb, bundle):
    features = extraire_caracteristiques(image_rgb).reshape(1, -1)

    class_names = bundle["class_names"]
    predictions = []

    for name, model in bundle["models"].items():
        label = int(model.predict(features)[0])

        score = None
        if hasattr(model, "decision_function"):
            score = float(
                np.asarray(model.decision_function(features)).ravel()[0]
            )

        predictions.append({
            "Model": name,
            "Prediction": class_names[label],
            "Label": label,
            "Decision score": score
        })

    return predictions