import streamlit as st
import joblib
import os
import math

# Set page config
st.set_page_config(page_title="TruthLens - Fake News Detector", page_icon="🔍", layout="centered")

# Custom CSS for UI mimicking the image
# Updated CSS to match TruthLens UI exactly
st.markdown("""
<style>
    /* Full Page Background */
    .stApp {
        background: radial-gradient(circle at center, #1e2235 0%, #11131f 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Hide Streamlit components */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}

    /* TruthLens Header Styling */
    .header-container {
        text-align: center;
        padding-top: 5rem;
        margin-bottom: 0.5rem;
    }
    .title-truth {
        font-size: 4rem;
        font-weight: 700;
        color: #ffffff;
    }
    .title-lens {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #9370DB, #da70d6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #8a8d9b;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Glassmorphism Container */
    .form-box {
        background: rgba(255, 255, 255, 0.03);
        padding: 40px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        max-width: 800px;
        margin: 0 auto;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #161826 !important;
        color: #ffffff !important;
        border: 1px solid #2a2d3e !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-size: 1rem !important;
        height: 200px !important;
    }

    /* Analyze Button Position and Style */
    div.stButton {
        display: flex;
        justify-content: flex-end;
        margin-top: 20px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4d76f1, #6b8cf5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        transition: 0.3s !important;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }

    /* Centered Footer */
    .footer-text {
        text-align: center;
        color: #5a5d72;
        font-size: 0.9rem;
        margin-top: 4rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and vectorizer at startup
@st.cache_resource
def load_models():
    model_path = 'fake_news_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return clf, vectorizer
    else:
        return None, None

clf, vectorizer = load_models()

if not clf or not vectorizer:
    st.error("Error: Model (`fake_news_model.joblib`) or vectorizer (`tfidf_vectorizer.joblib`) not found. Please train the model first.")
else:
    # Use HTML wrapper to mimic the dark container background
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    
    text_input = st.text_area("", height=200, placeholder="Paste article text here to verify its authenticity...", label_visibility="collapsed")
    
    submit_button = st.button("Analyze Content")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing content authenticity..."):
                try:
                    # Vectorize the input text
                    text_vectorized = vectorizer.transform([text_input])
                    prediction = clf.predict(text_vectorized)[0]
                    
                    # Calculate a pseudo-confidence score
                    dist = abs(clf.decision_function(text_vectorized)[0])
                    confidence = 1 / (1 + math.exp(-dist))
                    confidence_percent = round(confidence * 100, 2)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 0:
                            st.markdown('<div class="result-real">✓ REAL CONTENT</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-fake">✗ FAKE CONTENT</div>', unsafe_allow_html=True)
                            
                    with col2:
                        st.metric(label="Confidence Score", value=f"{confidence_percent}%")
                        st.progress(confidence)
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown('<div class="footer-text">Powered by LinearSVC & TF-IDF • IBM Project 2026</div>', unsafe_allow_html=True)

