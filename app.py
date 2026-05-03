import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- Chargement des modèles ---
tfidf = joblib.load('tfidf.joblib')
le    = joblib.load('label_encoder.joblib')
rf    = joblib.load('model_rf.joblib')
gb    = joblib.load('model_gb.joblib')

stop_words = set(stopwords.words('english'))

def nettoyer_texte(texte):
    if not texte:
        return ''
    texte = texte.lower()
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    mots  = [m for m in texte.split() if m not in stop_words]
    return ' '.join(mots)

# --- Interface ---
st.title('Fake News Type Detector')

modele_choisi = st.selectbox('Choisir le modèle', ['Random Forest', 'Gradient Boosting'])

titre = st.text_input('Titre de l article')
texte = st.text_area('Texte de l article', height=200)

if st.button('Prédire'):
    if not titre and not texte:
        st.warning('Entre au moins un titre ou un texte.')
    else:
        # Nettoyage + vectorisation
        texte_combined = nettoyer_texte(titre) + ' ' + nettoyer_texte(texte)
        X = tfidf.transform([texte_combined])

        # Prédiction
        if modele_choisi == 'Random Forest':
            prediction = rf.predict(X)[0]
            probabilites = rf.predict_proba(X)[0]
        else:
            prediction = gb.predict(X.toarray())[0]
            probabilites = gb.predict_proba(X.toarray())[0]

        label = le.inverse_transform([prediction])[0]

        st.success(f'Type prédit : **{label}**')

        # Afficher les probabilités
        st.subheader('Confiance par classe')
        for classe, proba in zip(le.classes_, probabilites):
            st.write(f'{classe} : {round(proba * 100, 1)}%')
            st.progress(float(proba))
