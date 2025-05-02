import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import langdetect
from langdetect import detect
import random

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar modelos
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocesamiento de texto
stop_words_en = set(stopwords.words('english'))
stop_words_es = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def preprocess(text, lang):
    tokens = word_tokenize(text.lower())
    if lang == 'es':
        stop_words = stop_words_es
    else:
        stop_words = stop_words_en
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def generate_recommendations(tweet, lang):
    recommendations = []
    if len(tweet.split()) < 5:
        recommendations.append("Haz tu tuit más largo agregando detalles." if lang == 'es' else "Make your tweet longer by adding details.")
    if "!" not in tweet:
        recommendations.append("Usa signos de exclamación para captar más atención." if lang == 'es' else "Use exclamation marks to grab more attention.")
    popular_emojis = ["🎉", "🔥", "💡", "😊", "💬"]
    if not any(emoji in tweet for emoji in popular_emojis):
        recommendations.append("Prueba agregar emojis como 🎉 para hacerlo más atractivo." if lang == 'es' else "Consider adding emojis like 🎉 to make it more engaging.")
    common_keywords_es = ["increíble", "nuevo", "gratis", "hoy", "tendencia"]
    common_keywords_en = ["amazing", "new", "free", "today", "trending"]
    common_keywords = common_keywords_es if lang == 'es' else common_keywords_en
    if not any(word in tweet.lower() for word in common_keywords):
        recommendations.append("Incluye palabras clave populares como 'gratis', 'nuevo' o 'hoy'." if lang == 'es' else "Include popular keywords like 'free', 'new', or 'today'.")
    return recommendations

# Interfaz Streamlit
st.title("🧠 Predicción de Tuits Virales con IA")
st.write("¡Hola! 👋 Soy tu asistente de predicción de tuits virales.")

user_name = st.text_input("Primero, ¿cómo te llamas? 😊")
if user_name:
    st.write(f"¡Encantado de conocerte, {user_name}! 🎉")

    st.write("Estoy aquí para ayudarte a predecir si tu tuit tiene potencial de ser viral.")
    tweet = st.text_input(f"¿Qué tuit tienes en mente, {user_name}? Escribe tu idea aquí:")
    if tweet:
        try:
            lang = detect(tweet)
        except langdetect.lang_detect_exception.LangDetectException:
            st.write("No pude detectar el idioma de tu tuit. Por favor, intenta de nuevo.")
            lang = 'en' 
            lang ='' # Por defecto, inglés

        processed_tweet = preprocess(tweet, lang)
        prediction = model.predict(vectorizer.transform([processed_tweet]))[0]
        confidence = model.predict_proba(vectorizer.transform([processed_tweet]))[0].max() * 100
        viral_text = "🔥 ¡Este tuit tiene potencial de ser viral!" if prediction else "💡 Quizás este tuit no se haga viral."
        st.markdown(f"### **Predicción:** {viral_text}")
        st.markdown(f"🔍 **Confianza del modelo:** {confidence:.2f}%")

        recommendations = generate_recommendations(tweet, lang)
        if recommendations:
            st.subheader(f"🌟 Recomendaciones para mejorar tu tuit, {user_name}:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write(f"¡Tu tuit ya es perfecto, {user_name}! 🎉")

        st.write("¿Quieres intentarlo con otro tuit? ¡Estoy aquí para ayudarte! 🚀")
