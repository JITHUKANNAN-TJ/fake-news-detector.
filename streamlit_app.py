import streamlit as st
import joblib
import os
import math

# Set page config
st.set_page_config(page_title="TruthLens - Fake News Detector", page_icon="🔍", layout="centered")

# Custom CSS for UI mimicking the image
st.markdown("""
<style>
    /* Full Page Background */
    .stApp {
        background-color: #1a1c29;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit components */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .css-15zrgzn {display: none}

    /* TruthLens Header Styling */
    .header-container {
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 0.5rem;
    }
    .title-truth {
        font-size: 3.5rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -1px;
    }
    .title-lens {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Chat/Form Container */
    .form-box {
        background-color: #1e2136;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
        border: 1px solid #2a2d45;
        margin-bottom: 2rem;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #161825 !important;
        color: #cbd5e1 !important;
        border: 1px solid #2a2d45 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1.05rem !important;
        min-height: 180px !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }
    .stTextArea label {
        display: none !important;
    }

    /* Analyze Button */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5, #6366f1, #818cf8);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
        display: block;
        margin-left: auto; /* Aligns button to the right */
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(90deg, #4338ca, #4f46e5, #6366f1);
    }
    div.stButton {
        display: flex;
        justify-content: flex-end;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 4rem;
        font-weight: 500;
    }
    
    /* Result styling */
    .result-real {
        color: #4ade80;
        font-weight: 700;
        font-size: 1.5rem;
        padding: 15px;
        background-color: rgba(74, 222, 128, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(74, 222, 128, 0.3);
        text-align: center;
        margin-top: 1rem;
    }
    .result-fake {
        color: #f87171;
        font-weight: 700;
        font-size: 1.5rem;
        padding: 15px;
        background-color: rgba(248, 113, 113, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(248, 113, 113, 0.3);
        text-align: center;
        margin-top: 1rem;
    }
    .stProgress .st-bo {
        background-color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# Main Title and Subtitle
st.markdown("""
<div class="header-container">
    <span class="title-truth">Truth</span><span class="title-lens">Lens</span>
</div>
<div class="subtitle">Next-Generation AI Authentication Engine</div>
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
