"""
Flower Recognition — Aplikasi Streamlit
"""

import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flower Recognition",
    page_icon="🌸",
    layout="centered"
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
MODEL_PATH = os.path.join(OUT_DIR, "final_flower_cnn_model.keras")
CLASS_PATH = os.path.join(OUT_DIR, "class_names.json")

IMG_SIZE = (160, 160)
CONF_THRESHOLD = 60

EMOJI = {
    "daisy": "🌼",
    "dandelion": "🌻",
    "rose": "🌹",
    "sunflower": "🌻",
    "tulip": "🌷",
}

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(
            135deg, #ff9a9e 0%, #fad0c4 50%, #fbc2eb 100%
        );
        padding: 2rem 1.5rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,.12);
    }

    .hero h1 {
        margin: 0;
        color: #5a1a3a;
        font-size: 2.2rem;
    }

    .hero p {
        margin: .4rem 0 0;
        color: #7a3a5a;
    }

    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(102,126,234,.35);
    }

    .result-card .emoji {
        font-size: 3rem;
    }

    .result-card .name {
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: capitalize;
    }

    .result-card .conf {
        font-size: 1.1rem;
        opacity: .9;
    }
    </style>

    <div class="hero">
        <h1>🌸 Flower Recognition</h1>
        <p>Upload foto bunga, AI akan menebak jenisnya dalam hitungan detik.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH) as f:
        class_names = json.load(f)

    return model, class_names


if not (
    os.path.exists(MODEL_PATH)
    and os.path.exists(CLASS_PATH)
):
    st.error(
        "Model belum ditemukan. Jalankan dulu notebook "
        "`flower_recognition.ipynb` sampai selesai agar "
        "folder `output/` terisi."
    )
    st.stop()

model, class_names = load_assets()


def predict(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)

    arr = np.expand_dims(
        np.array(img, dtype="float32"),
        axis=0
    )

    probs = model.predict(arr, verbose=0)[0]

    return probs


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Tentang Model")

    st.write("**Arsitektur:** MobileNetV2 (transfer learning)")
    st.write(f"**Input:** {IMG_SIZE[0]}×{IMG_SIZE[1]} piksel")
    st.write(f"**Jumlah kelas:** {len(class_names)}")

    st.markdown("**Kelas yang dikenali:**")

    for c in class_names:
        st.write(f"{EMOJI.get(c, '')} {c.capitalize()}")

    st.divider()

    st.caption(
        "Tips: gunakan foto dengan bunga yang jelas "
        "dan pencahayaan cukup."
    )


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_pred, tab_eval = st.tabs(
    ["🔍 Prediksi", "📊 Evaluasi Model"]
)

# ------------------------------------------------------------
# TAB PREDIKSI
# ------------------------------------------------------------
with tab_pred:

    mode = st.radio(
        "Sumber gambar",
        ["Upload", "Kamera"],
        horizontal=True
    )

    if mode == "Upload":
        file = st.file_uploader(
            "Pilih gambar bunga",
            type=["jpg", "jpeg", "png", "webp"]
        )
    else:
        file = st.camera_input("Ambil foto bunga")

    if file is not None:

        image = Image.open(file)
        probs = predict(image)

        order = np.argsort(probs)[::-1]

        top = order[0]
        name = class_names[top]
        conf = float(probs[top]) * 100

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(
                image,
                caption="Gambar input",
                use_container_width=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="emoji">
                        {EMOJI.get(name, '')}
                    </div>
                    <div class="name">{name}</div>
                    <div class="conf">
                        Keyakinan: {conf:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if conf < CONF_THRESHOLD:
            st.warning(
                "Keyakinan rendah. Coba foto yang lebih jelas "
                "atau dari sudut lain."
            )
        else:
            st.success("Prediksi cukup meyakinkan.")

        st.subheader("Probabilitas semua kelas")

        for i in order:
            p = float(probs[i])

            st.write(
                f"{EMOJI.get(class_names[i], '')} "
                f"**{class_names[i].capitalize()}** — "
                f"{p * 100:.1f}%"
            )

            st.progress(min(max(p, 0.0), 1.0))

    else:
        st.info(
            "Belum ada gambar. Upload foto atau gunakan "
            "kamera untuk mulai."
        )


# ------------------------------------------------------------
# TAB EVALUASI MODEL
# ------------------------------------------------------------
with tab_eval:

    st.write(
        "Hasil evaluasi yang dihasilkan notebook "
        "pada data validasi."
    )

    figs = [
        ("2_training_curves.png", "Grafik Akurasi & Loss"),
        ("3_confusion_matrix.png", "Confusion Matrix"),
        ("4_sample_predictions.png", "Contoh Prediksi"),
    ]

    found = False

    for fname, title in figs:

        path = os.path.join(OUT_DIR, fname)

        if os.path.exists(path):
            found = True

            st.subheader(title)
            st.image(
                path,
                use_container_width=True
            )

    if not found:
        st.info(
            "Gambar evaluasi belum ada di folder `output/`."
        )


st.divider()

st.caption("Dibuat dengan TensorFlow & Streamlit 🌸")