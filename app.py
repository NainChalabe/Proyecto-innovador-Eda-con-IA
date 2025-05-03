import streamlit as st
import pickle
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
from textblob import TextBlob

# Descargar recursos necesarios
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar historial en sesión
if 'sentiments' not in st.session_state:
    st.session_state.sentiments = []
if 'length_vs_confidence' not in st.session_state:
    st.session_state.length_vs_confidence = []
if 'confidence_data' not in st.session_state:
    st.session_state.confidence_data = []

# Cargar modelos
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError as e:
    st.error(f"Error al cargar modelos: {e}")
    st.stop()

# Preprocesamiento
stop_words_en = set(stopwords.words('english'))
stop_words_es = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positivo", polarity
    elif polarity < -0.1:
        return "Negativo", polarity
    else:
        return "Neutral", polarity

def preprocess(text, lang):
    stop_words = stop_words_es if lang == 'es' else stop_words_en
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tweet_length = len(text.split())
    sentiment_category, sentiment_value = analyze_sentiment(text)
    return ' '.join(tokens), tweet_length, sentiment_category, sentiment_value

def generate_recommendations(tweet, lang='es'):
    recs = []
    if len(tweet.split()) < 5:
        recs.append("Haz tu tuit más largo agregando detalles.")
    if "!" not in tweet:
        recs.append("Usa signos de exclamación para captar más atención.")
    if not any(emoji in tweet for emoji in ["🎉", "🔥", "💡", "😊", "💬"]):
        recs.append("Prueba agregar emojis como 🎉 para hacerlo más atractivo.")
    if not any(word in tweet.lower() for word in ["increíble", "nuevo", "gratis", "hoy", "tendencia"]):
        recs.append("Incluye palabras clave populares como 'gratis', 'nuevo' o 'hoy'.")
    return recs

# Interfaz
st.title("Predicción de Tuits Virales con IA")
st.write("¡Hola! Soy tu asistente de predicción de tuits virales. ")

user_name = st.text_input("¿Cómo te llamas?")
if user_name:
    st.write(f"¡Encantado, {user_name}! Escribe tu tuit y veamos qué tan viral puede ser. ")
    tweet = st.text_input("Escribe tu tuit aquí:")

    if tweet:
        try:
            lang = detect(tweet)
        except LangDetectException:
            lang = 'es'

        processed_tweet, length, sentiment_cat, sentiment_val = preprocess(tweet, lang)
        vector = vectorizer.transform([processed_tweet])
        confidence = model.predict_proba(vector)[0].max() * 100
        prediction = model.predict(vector)[0]

        viral_text = "🔥 ¡Este tuit tiene potencial de ser viral!" if confidence >= 75 else "💡 Este tuit tiene un bajo potencial de ser viral."
        st.markdown(f"### **Predicción:** {viral_text}")
        st.markdown(f"**Confianza del modelo:** {confidence:.2f}%")

        recs = generate_recommendations(tweet, lang)
        if recs:
            st.subheader("Recomendaciones para mejorar tu tuit:")
            for r in recs:
                st.write(f"- {r}")
        else:
            st.write("¡Tu tuit ya es excelente!")

        st.write(f"**Longitud del tuit:** {length} palabras")
        st.write(f"**Sentimiento del tuit:** {sentiment_cat} (valor: {sentiment_val:.2f})")

        # Guardar en sesión
        st.session_state.sentiments.append(sentiment_cat)
        st.session_state.length_vs_confidence.append((length, confidence))
        st.session_state.confidence_data.append({'Sentimiento': sentiment_cat, 'Confianza': confidence})

        # Gráfica 1: Longitud vs Confianza
        st.subheader(" Longitud del tuit vs Confianza de viralidad")
        df1 = pd.DataFrame(st.session_state.length_vs_confidence, columns=["Longitud", "Confianza"])
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df1, x="Longitud", y="Confianza", s=100, color='blue', ax=ax1)
        ax1.set_xlabel("Longitud del tuit")
        ax1.set_ylabel("Confianza (%)")
        ax1.set_title("Relación entre Longitud y Confianza")
        st.pyplot(fig1)

        # Gráfica 2: Confianza promedio por Sentimiento
        st.subheader("Confianza promedio de viralidad por sentimiento")
        df2 = pd.DataFrame(st.session_state.confidence_data)
        avg_conf = df2.groupby("Sentimiento")["Confianza"].mean().reset_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(data=avg_conf, x="Sentimiento", y="Confianza", palette="coolwarm", ax=ax2)
        ax2.set_title("Confianza Promedio de Viralidad por Sentimiento")
        ax2.set_ylabel("Confianza Promedio (%)")
        ax2.set_xlabel("Sentimiento")
        ax2.set_ylim(0, 100)
        st.pyplot(fig2)

