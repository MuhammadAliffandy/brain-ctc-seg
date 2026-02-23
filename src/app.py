import streamlit as st
from PIL import Image
import sys
import os

# Add the parent directory to sys.path to allow importing from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.predictor import predict_segmentation
from src.services.llm import get_explanation

st.set_page_config(
    page_title="Brain CTC Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalist Clinical SaaS Look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #334155 !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Buttons */
    .stButton>button {
        background-color: #0284C7;
        color: #FFFFFF !important;
        border-radius: 6px;
        border: 1px solid #0284C7;
        padding: 10px 24px;
        font-weight: 500;
        transition: none !important; /* No hover animation */
        box-shadow: none !important;
    }
    .stButton>button:hover {
        background-color: #0369A1 !important;
        border-color: #0369A1 !important;
        color: #FFFFFF !important;
    }
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        border: 1px dashed #CBD5E1 !important;
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
    }
    /* Result containers */
    .result-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    /* Streamlit Alerts */
    .stAlert {
        background-color: #F0F9FF !important;
        color: #0369A1 !important;
        border: 1px solid #BAE6FD !important;
        border-left: 4px solid #0284C7 !important;
        border-radius: 4px !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    /* Hide default icons in alerts */
    .stAlert svg {
        display: none;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #0284C7 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("Analysis Settings")
    
    st.markdown("### Patient Information")
    patient_id = st.text_input("Patient ID / Case Number", placeholder="e.g., PT-2026-001")
    
    st.markdown("---")
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
st.title("Brain Segmentation AI")
st.markdown("### Clinical Support System for Automated Segmentation")
st.markdown("Upload a brain scan image to detect and segment regions of interest using our advanced AI model.")

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display Image Metadata
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
    st.caption(f"Uploaded: {file_details['Filename']} ({file_details['FileSize']})")

    # Perform Prediction
    try:
        with st.spinner("Processing image..."):
            original_image, result_mask = predict_segmentation(uploaded_file, modality)
        
        # Use Tabs for better organization
        tab1, tab2 = st.tabs(["Visual Analysis", "Clinical Report"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Original Scan")
                st.image(original_image, use_container_width=True)
                
            with col2:
                st.subheader("Segmentation Result")
                st.image(result_mask, use_container_width=True, clamp=True, channels="L")
                
        with tab2:
            with st.container():
                st.subheader("AI Analysis Report")
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
            m1.metric("Patient ID", patient_id if patient_id else "Anonymous")
            m2.metric("Modality", modality)
            m3.metric("Confidence", f"{confidence_threshold * 100:.0f}%")
            
            st.markdown("---")
            
            # Prepare data for LLM
            detected_regions = "Lesion detected in the central region (Simulated)."
            
            # Use session state to cache the explanation so it doesn't regenerate on download
            session_key = f"explanation_{uploaded_file.name}_{modality}_{confidence_threshold}"
            if session_key not in st.session_state:
                with st.spinner("Generating AI Interpretation..."):
                    st.session_state[session_key] = get_explanation(modality, confidence_threshold, detected_regions)
            
            explanation = st.session_state[session_key]
                
            st.markdown(f"**Clinical Interpretation:**")
            st.info(explanation)
            
            # Download Report Feature
            import textwrap
            report_content = (
                "BRAIN SEGMENTATION AI - CLINICAL REPORT\n"
                "---------------------------------------\n"
                f"Patient ID  : {patient_id if patient_id else 'Anonymous'}\n"
                f"Modality    : {modality}\n"
                f"Confidence  : {confidence_threshold}\n"
                f"File        : {file_details['Filename']}\n"
                "\n"
                "INTERPRETATION:\n"
                f"{explanation}\n"
                "\n"
                "Note: This is an AI-generated report and should be verified by a medical professional.\n"
            )
            
            st.download_button(
                label="Download Report (TXT)",
                data=report_content,
                file_name=f"report_{patient_id if patient_id else 'anon'}.txt",
                mime="text/plain"
            )
            
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
