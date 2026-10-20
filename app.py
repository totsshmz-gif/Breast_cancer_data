import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

st.set_page_config(
    page_title="Prédiction Cancer du Sein",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('breast_cancer_model.keras')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_models()

st.title("🩺 Prédiction de Tumeur du Sein (Benign ou Malignant)")
st.write("""
Cette application utilise un réseau de neurones (Deep Learning) entrainé sur le jeu de données **Breast Cancer Wisconsin**. 
Modifiez les valeurs des caractéristiques ci-dessous pour prédire la nature de la tumeur.
""")

st.divider()

st.subheader("Entrez les caractéristiques de la tumeur")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean (Rayon moyen)", value=14.12, format="%.4f")
    texture_mean = st.number_input("Texture Mean", value=19.28, format="%.4f")
    perimeter_mean = st.number_input("Perimeter Mean (Périmètre)", value=91.96, format="%.4f")
    area_mean = st.number_input("Area Mean (Aire)", value=654.8, format="%.4f")
    smoothness_mean = st.number_input("Smoothness Mean", value=0.0963, format="%.4f")

with col2:
    compactness_mean = st.number_input("Compactness Mean", value=0.1043, format="%.4f")
    concavity_mean = st.number_input("Concavity Mean", value=0.0888, format="%.4f")
    concave_points_mean = st.number_input("Concave Points Mean", value=0.0489, format="%.4f")
    symmetry_mean = st.number_input("Symmetry Mean", value=0.1811, format="%.4f")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.0628, format="%.4f")

features_completes = [
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
    0.35, 1.25, 2.45, 35.0, 0.007, 0.025, 0.032, 0.012, 0.020, 0.003,
    16.25, 25.60, 107.5, 880.0, 0.13, 0.25, 0.27, 0.11, 0.29, 0.08
]

if st.button("Lancer la Prédiction", type="primary"):
    input_data = np.array([features_completes])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    probabilite = float(prediction[0][0])
    
    st.divider()
    st.subheader("Résultat de l'analyse")
    
    if probabilite > 0.5:
        st.error(f"⚠️ Résultat : **Maligne (Malignant)**\n\nProbabilité : {probabilite * 100:.2f}%")
    else:
        st.success(f"✅ Résultat : **Bénigne (Benign)**\n\nProbabilité : {(1 - probabilite) * 100:.2f}%")
