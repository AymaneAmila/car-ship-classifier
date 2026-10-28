# car-ship-classifier

# Car/Ship Image Classifier

This Streamlit app reuses the fitted six models from the supplied notebook.
It expects the same 59 handcrafted features in the same order as `feature_extraction.py`.

## 1. Export the model from the notebook

Run this in a new notebook cell **after the models have successfully fitted **:

```python
from pathlib import Path
import joblib
import sklearn

export_dir = Path("artifacts")
export_dir.mkdir(parents=True, exist_ok=True)

bundle = {
    "models": modeles,
    "class_names": {0: "ship", 1: "car"},
    "feature_count": 71,
    "test_metrics": resultats_details,
    "sklearn_version": sklearn.__version__
}

model_path = export_dir / "six_models.joblib"
joblib.dump(bundle, model_path)

print(f"Six models saved to: {model_path.resolve()}")
```


Copy the resulting `.joblib` file into this project's `artifacts/` folder.
Do not commit the dataset or the original training notebook unless you intend to share them.

## 2. Install and run locally

Use the same Python/scikit-learn version as the environment that created the model
where possible. `joblib` files are Python serialization artifacts; only load model
files you trust.

Then:
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## 3. Unknown-image limitation

The app displays a warning that the model only learned `car` and `ship`. By default,
`REJECTION_THRESHOLD = None`, because the notebook does not establish a defensible
unknown-image threshold. The SVM decision score is not a calibrated probability,
and unrelated images can have large-magnitude scores too.

To enable a margin-based rejection rule, validate a threshold on a separate validation
set containing representative cars, ships, and unrelated images. Then set
`REJECTION_THRESHOLD` in `app.py` to the selected threshold. A low absolute decision
score indicates proximity to the binary decision boundary, not necessarily that an
image is unrelated.

