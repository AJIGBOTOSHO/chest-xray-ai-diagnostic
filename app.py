import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

# ─────────────────────────────────────────────
# CONFIG – EXACTLY AS DURING TRAINING
# ─────────────────────────────────────────────
MODEL_PATH = "densenet121_nigeria.keras"
CLASS_LABELS = ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID-19']
IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Chest X‑ray AI",
    page_icon="🫁",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0f2027; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main { background-color: #f5f7fa; }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .top-pred {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .confidence-label { font-size: 0.85rem; color: #666; margin-bottom: 0.2rem; }
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #856404;
    }
    h1 { color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING (build Grad‑CAM dependencies)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_gradcam():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found.")
        st.stop()
    model = load_model(MODEL_PATH, compile=False)

    # --- Rebuild DenseNet121 base without pooling (keeps spatial dims) ---
    trained_densenet = model.get_layer('densenet121')
    base_weights = trained_densenet.get_weights()

    new_base = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
        pooling=None                     # output: (7, 7, 1024)
    )
    new_base.set_weights(base_weights)

    # Classifier layers = everything after DenseNet121 in the Sequential
    classifier_layers = model.layers[2:]  # GAP, BN, Dense, Dropout, …

    return model, new_base, classifier_layers

# ─────────────────────────────────────────────
# PREPROCESSING (CLAHE + normalise)
# ─────────────────────────────────────────────
def apply_clahe(img_array):
    """CLAHE enhancement – exactly as in training."""
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb.astype(np.float32) / 255.0

def preprocess(image: Image.Image) -> np.ndarray:
    """Resize → CLAHE → normalise → add batch dim."""
    img = image.convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    processed = apply_clahe(img_array)
    return np.expand_dims(processed, axis=0)

# ─────────────────────────────────────────────
# GRAD‑CAM (single forward pass, no graph errors)
# ─────────────────────────────────────────────
def make_gradcam_heatmap(img_array, new_base, classifier_layers, pred_index):
    with tf.GradientTape() as tape:
        conv_features = new_base(img_array, training=False)     # (1,7,7,1024)
        tape.watch(conv_features)
        x = conv_features
        for layer in classifier_layers:
            x = layer(x, training=False)
        predictions = x                                         # (1,4)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (1024,)
    conv_out = conv_features[0]                                 # (7,7,1024)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    max_val = heatmap.max()
    if max_val == 0:
        return None
    heatmap = heatmap / max_val
    return heatmap

# ─────────────────────────────────────────────
# BOUNDING BOX FROM HEATMAP
# ─────────────────────────────────────────────
def heatmap_to_bbox(heatmap, original_shape, threshold=0.5):
    h_orig, w_orig = original_shape[0], original_shape[1]
    heatmap_big = cv2.resize(heatmap, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    heatmap_big = (heatmap_big - heatmap_big.min()) / (heatmap_big.max() - heatmap_big.min() + 1e-10)
    _, binary = cv2.threshold(
        np.uint8(255 * heatmap_big), int(threshold * 255), 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

# ─────────────────────────────────────────────
# OVERLAY (heatmap + bounding box) — RGB safe
# ─────────────────────────────────────────────
def create_overlay(original_pil, heatmap, pred_class, confidence, threshold=0.5):
    # ── FIX: always work in RGB numpy arrays ──
    original = np.array(original_pil.convert("RGB"))
    h, w, _ = original.shape

    heatmap_big = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_big = (heatmap_big - heatmap_big.min()) / (heatmap_big.max() - heatmap_big.min() + 1e-10)

    # applyColorMap returns BGR → convert to RGB immediately
    heatmap_bgr   = cv2.applyColorMap(np.uint8(255 * heatmap_big), cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.6, heatmap_rgb, 0.4, 0)

    bbox = heatmap_to_bbox(heatmap, original.shape, threshold)
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(overlay, f"{pred_class} ({confidence:.2%})",
                    (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return Image.fromarray(overlay), bbox   # already RGB → PIL safe

# ─────────────────────────────────────────────
# CONFIDENCE BAR CHART
# ─────────────────────────────────────────────
def plot_confidence(probs, labels, top_idx):
    colors = ["#e63946" if i == top_idx else "#457b9d" for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(labels, probs * 100, color=colors, height=0.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=72)
    st.markdown("## ⚙️ Settings")
    show_gradcam = st.toggle("Show Grad‑CAM + Bounding Box", value=True)
    gradcam_alpha = st.slider("Heatmap opacity", 0.2, 0.8, 0.4, step=0.05,
                              disabled=not show_gradcam)
    bbox_threshold = st.slider("Bounding box threshold", 0.3, 0.8, 0.5, step=0.05,
                               disabled=not show_gradcam)
    st.markdown("---")
    st.markdown("**Model:** DenseNet121 (fine-tuned)")
    st.markdown(f"**Classes:** {', '.join(CLASS_LABELS)}")
    st.markdown("**Input size:** 224 × 224")
    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'>⚠️ For research use only. "
        "Not a substitute for clinical diagnosis.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("🫁 Explainable AI Diagnostic Assistant")
st.caption("DenseNet121 · Trained on Nigerian patient data · 4‑class classification")

uploaded = st.file_uploader(
    "Upload a chest X‑ray image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG and PNG formats",
)

if uploaded:
    model, new_base, classifier_layers = load_model_and_gradcam()

    # ── FIX 1: .copy() forces full load into memory on cloud ──
    image = Image.open(uploaded).copy()

    col_img, col_results = st.columns([1, 1], gap="large")

    # ── Left: original image ──────────────────
    with col_img:
        st.subheader("Uploaded X‑Ray")
        st.image(np.array(image.convert("RGB")), use_column_width=True, caption=str(uploaded.name))
        # st.image(image, use_container_width=True, caption=uploaded.name)

    # ── Inference ────────────────────────────
    with st.spinner("Analysing…"):
        img_array = preprocess(image)
        probs = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_label = CLASS_LABELS[top_idx]
        top_conf = float(probs[top_idx])

    # ── Right: results ────────────────────────
    with col_results:
        st.subheader("Prediction")
        st.markdown(
            f"<div class='result-card'>"
            f"<div class='confidence-label'>Top prediction</div>"
            f"<div class='top-pred'>{top_label}</div>"
            f"<div style='font-size:1.3rem; color:#457b9d; margin-top:4px;'>"
            f"{top_conf * 100:.1f}% confidence</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Class probabilities**")
        fig = plot_confidence(probs, CLASS_LABELS, top_idx)
        st.pyplot(fig)
        plt.close(fig)

    # ── Grad‑CAM & Bounding Box ──────────────
    if show_gradcam:
        st.markdown("---")
        st.subheader("🔍 Explainability (Grad‑CAM + Bounding Box)")
        st.caption("Highlighted regions show where the model focused.")

        with st.spinner("Generating explanation…"):
            heatmap = make_gradcam_heatmap(img_array, new_base, classifier_layers, top_idx)

        if heatmap is not None:
            overlay_img, bbox = create_overlay(
                image, heatmap, top_label, top_conf, threshold=bbox_threshold
            )

            # ── FIX 2: heatmap_viz converted BGR→RGB before st.image ──
            heatmap_bgr = cv2.applyColorMap(
                np.uint8(255 * cv2.resize(heatmap, IMG_SIZE)), cv2.COLORMAP_JET
            )
            heatmap_viz = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image,caption="Original",                use_column_width=True)
            with col2:
                st.image(overlay_img, caption="Grad‑CAM + Bounding Box", use_column_width=True)
            with col3:
                st.image(heatmap_viz, caption="Grad‑CAM Heatmap",        use_column_width=True)

            # --- AI Explanation Summary ---
            notes = {
                'Normal':      'No acute cardiopulmonary abnormality detected.',
                'Pneumonia':   'Consolidation or diffuse opacities possible.',
                'Tuberculosis':'Apical infiltrates / cavitary lesions typical.',
                'COVID-19':    'Peripheral ground‑glass opacities common.'
            }
            if bbox:
                x, y, bw, bh = bbox
                original_arr  = np.array(image.convert("RGB"))
                h_orig, w_orig, _ = original_arr.shape
                region = "upper" if y < h_orig * 0.33 else ("middle" if y < h_orig * 0.66 else "lower")
                side   = "right" if x > w_orig / 2 else "left"
                loc_text = f"The model focused on the **{region} {side}** lung field."
            else:
                loc_text = "No distinct focal region identified."

            st.markdown("### 🔍 AI Explanation Summary")
            st.markdown(f"""
| Item | Value |
|------|-------|
| **Predicted Condition** | {top_label} |
| **Confidence** | {top_conf:.2%} |
| **Localisation** | {loc_text} |
| **Clinical Note** | {notes[top_label]} |
| **Recommendation** | Radiologist review and confirmatory testing (GeneXpert, PCR) advised. |
""")
        else:
            st.warning("Grad‑CAM could not be generated for this model architecture.")

    # ── Download report ───────────────────────
    st.markdown("---")
    result_text = (
        f"Chest X‑Ray Analysis Report\n"
        f"============================\n"
        f"File       : {uploaded.name}\n"
        f"Prediction : {top_label}\n"
        f"Confidence : {top_conf * 100:.2f}%\n\n"
        f"All class probabilities:\n"
        + "\n".join(f"  {lbl:<16}: {p*100:.2f}%" for lbl, p in zip(CLASS_LABELS, probs))
        + "\n\n⚠️ For research use only — not a clinical diagnosis."
    )
    st.download_button(
        "⬇️ Download Report",
        data=result_text,
        file_name="xray_report.txt",
        mime="text/plain",
    )

else:
    st.info("👆 Upload a chest X‑ray image to get started.")
    st.markdown("""
    **What this tool does:**
    - Classifies chest X‑rays into 4 categories: **Normal · Pneumonia · Tuberculosis · COVID-19**
    - Generates **Grad‑CAM heatmaps** and **bounding‑box localisation**
    - Provides an **AI explanation summary** and downloadable report
    """) 