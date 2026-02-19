import streamlit as st
from PIL import Image
import sys
import os

# Add the parent directory to sys.path to allow importing from backend
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.predictor import predict_segmentation
from src.services.llm import get_explanation

st.set_page_config(
    page_title="Brain CTC Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Clinical Environment Look
st.markdown("""
<style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #005F9E;
    }
    h1 {
        color: #003B5C;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h2, h3 {
        color: #0072B5;
    }
    .uploaded-file {
        border: 2px dashed #0072B5;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .result-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("Settings")
    
    st.markdown("### Modality Selection")
    modality = st.radio(
        "Choose Scan Type:",
        ("CT Scan", "CTC", "Combined (CT + CTC)")
    )
    
    st.markdown("---")
    st.markdown("### Model Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    st.info("Ensure the uploaded image is in a supported format (PNG, JPG, JPEG).")

# Main Content
st.title("🧠 Brain Segmentation AI")
st.markdown("### Clinical Support System for Automated Segmentation")
st.markdown("Upload a brain scan image to detect and segment regions of interest using our advanced AI model.")

col1, col2 = st.columns([1, 1])

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Perform Prediction
    try:
        original_image, result_mask = predict_segmentation(uploaded_file, modality)
        
        with col1:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Original Scan")
            st.image(original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Segmentation Result")
            st.image(result_mask, use_container_width=True, clamp=True, channels="L")
            st.success("Segmentation completed successfully.")
            st.markdown('</div>', unsafe_allow_html=True)
            
# Analysis Report
        st.markdown("---")
        st.subheader("🔍 AI Analysis Report")
        
        # Prepare data for LLM
        detected_regions = "Lesion detected in the central region (Simulated)."
        
        with st.spinner("Generating AI Interpretation..."):
            explanation = get_explanation(modality, confidence_threshold, detected_regions)
            
        st.markdown(f"**Clinical Interpretation:**")
        st.info(explanation)
        
        st.write(f"**Technical Details:**")
        st.write(f"- **Modality:** {modality}")
        st.write(f"- **Confidence Score:** {confidence_threshold}")
        
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("Please upload an image to start the analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>Brain CTC Segmentation &copy; 2026 | Enhanced Clinical AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
