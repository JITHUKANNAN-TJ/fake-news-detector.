import streamlit as st
import joblib
import os
import math

# 1. Page Configuration
st.set_page_config(page_title="TruthLens AI", page_icon="🔍", layout="centered")

# 2. Custom CSS for the "TruthLens" Dark Aesthetic
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, #1e2235 0%, #11131f 100%);
    }
    header, footer, [data-testid="stHeader"] {visibility: hidden;}

    .header-container { text-align: center; padding-top: 3rem; }
    .title-truth { font-size: 3.5rem; font-weight: 700; color: #ffffff; }
    .title-lens { 
        font-size: 3.5rem; font-weight: 700; 
        background: linear-gradient(90deg, #9370DB, #da70d6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; color: #8a8d9b; font-size: 1rem; margin-bottom: 2rem; }

    /* The Glassmorphism Container */
    .main-box {
        background: rgba(255, 255, 255, 0.03);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #161826 !important;
        color: #ffffff !important;
        border: 1px solid #2a2d3e !important;
        border-radius: 12px !important;
        height: 180px !important;
    }

    /* Button Alignment */
    div.stButton { display: flex; justify-content: flex-end; margin-top: 15px; }
    div.stButton > button {
        background: linear-gradient(90deg, #4d76f1, #6b8cf5) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading Logic
@st.cache_resource
def load_assets():
    model_path = 'fake_news_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None

clf, vectorizer = load_assets()

# 4. Header UI
st.markdown('<div class="header-container"><span class="title-truth">Truth</span><span class="title-lens">Lens</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Generation AI Authentication Engine</div>', unsafe_allow_html=True)

# 5. Main Interaction Logic
if not clf or not vectorizer:
    st.error("Model files missing! Please run your training script first.")
else:
    with st.container():
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        
        user_input = st.text_area("", placeholder="Paste article text here to verify its authenticity...", label_visibility="collapsed")
        
        if st.button("Analyze Content"):
            if user_input.strip():
                # Prediction Logic
                vec_text = vectorizer.transform([user_input])
                prediction = clf.predict(vec_text)[0]
                
                # Confidence Calculation
                dist = abs(clf.decision_function(vec_text)[0])
                confidence = 1 / (1 + math.exp(-dist))
                
                # Result Display
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 0:
                        st.success("✅ REAL CONTENT")
                    else:
                        st.error("🚨 FAKE CONTENT")
                with col2:
                    st.metric("Confidence Score", f"{round(confidence * 100, 2)}%")
                    st.progress(confidence)
            else:
                st.warning("Please enter text before analyzing.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# 6. Footer
st.markdown('<br><div style="text-align: center; color: #5a5d72; font-size: 0.8rem;">Powered by LinearSVC & TF-IDF • IBM Project 2026</div>', unsafe_allow_html=True)
