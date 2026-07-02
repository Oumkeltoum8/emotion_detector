import streamlit as st
from src.predict import predict_emotion

st.set_page_config(page_title="Détecteur d'Émotions", page_icon="🎭")
st.title(" Détecteur d'Émotions Multiclasse")
st.write("Entrez un texte en anglais pour détecter l'émotion dominante")

text = st.text_area("Votre texte ici...", height=100)

if st.button(" Analyser"):
  if text.strip():
    emotion, confidence = predict_emotion(text)
    st.success(f"Émotion détectée : **{emotion.upper()}**")
    st.metric("Score de confiance", f"{confidence:.1f}%")
  else:
    st.warning("Veuillez entrer un texte")

# Lancer avec : streamlit run app.py