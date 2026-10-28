import io
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from inference import load_bundle, predict_with_all_models

st.set_page_config(page_title="Six-Model Image Classifier", page_icon="🖼️", layout="centered")

@st.cache_resource
def get_model_bundle():
    return load_bundle()

st.title("Car vs Ship — Six-Model Comparison")
st.write(
    "Upload one image to compare the predictions of all six classifiers trained "
    "in the notebook: three SVM kernels, two KNN configurations, and a decision tree."
)
st.info(
    "The models only know the classes car and ship. An unrelated image may still "
    "be assigned to one of these classes. A single image comparison shows model "
    "agreement, not overall accuracy or proof that the image belongs to either class."
)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        image.load()
        image = image.convert("RGB")
        image_rgb = np.asarray(image, dtype=np.uint8)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Compare all six models", type="primary", use_container_width=True):
            with st.spinner("Extracting features and running all six models..."):
                bundle = get_model_bundle()
                results = predict_with_all_models(image_rgb, bundle)

            counts = Counter(item["Prediction"] for item in results)
            max_votes = max(counts.values())
            majority = [name for name, votes in counts.items() if votes == max_votes]

            st.subheader("Predictions")
            display_rows = []
            for item in results:
                display_rows.append({
                    "Model": item["Model"],
                    "Prediction": item["Prediction"].title(),
                })
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

            if len(majority) == 1:
                st.success(f"Majority prediction: **{majority[0].title()}** ({max_votes}/6 models)")
            else:
                st.warning(
                    f"No unique majority: {max_votes}/6 votes each for "
                    + " and ".join(name.title() for name in majority)
                )

            st.subheader("Prediction distribution")
            st.bar_chart(pd.Series(counts, name="Number of models"))

            st.caption(
                "SVM decision scores are raw decision-function values, not probabilities. "
                "They are not directly comparable across different kernels. A majority vote "
                "is a useful summary but is not a calibrated confidence score."
            )

            if "test_metrics" in bundle:
                with st.expander("Notebook test-set metrics (separate from this image)"):
                    metrics = pd.DataFrame(bundle["test_metrics"]).T
                    st.dataframe(metrics, use_container_width=True)
                    st.caption(
                        "These metrics were calculated on the notebook's held-out test set. "
                        "They do not measure the correctness of this individual uploaded image."
                    )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process this image: {exc}")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")

st.divider()
st.caption("Educational demonstration. Use a separate labelled test set to compare model performance reliably.")
