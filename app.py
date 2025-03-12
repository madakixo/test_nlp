import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import faiss
import os
import time

# Define device (CPU for deployment simplicity)
device = torch.device("cpu")
st.write(f"✅ Using device: {device}")

# Define paths (adjust based on deployment environment)
model_folder = "model_classifier"  # Local folder or full path
data_path = "processed_diseases-priority"    # Local file or full path

# Define the Optimized Disease Classifier (must match training script)
class OptimizedDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OptimizedDiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load model and artifacts
@st.cache_resource
def load_model_and_artifacts(model_folder, input_dim=5000, num_classes=50):
    try:
        model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(model_folder, "best_model.pth"), map_location=device))
        model.eval()
        
        with open(os.path.join(model_folder, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(model_folder, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        faiss_index = faiss.read_index(os.path.join(model_folder, "faiss_index.bin"))
        
        df = pd.read_csv(data_path)
        df["Disease"] = label_encoder.transform(df["Disease"])
        
        # Extract common symptoms for suggestions
        all_symptoms = ','.join(df["Symptoms"]).split(',')
        symptom_list = sorted(set([s.strip().lower() for s in all_symptoms if s.strip()]))
        
        return model, tfidf, label_encoder, faiss_index, df, symptom_list
    except Exception as e:
        st.error(f"Error loading model or artifacts: {e}")
        return None, None, None, None, None, None

# Prediction function with confidence
def predict(symptoms, model, tfidf, label_encoder, faiss_index, df):
    try:
        symptoms_tfidf = tfidf.transform([symptoms]).toarray()
        symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(symptoms_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item() * 100
        disease_name = label_encoder.inverse_transform([predicted_class])[0]
        
        disease_info = df[df["Disease"] == predicted_class].iloc[0]
        treatment = disease_info["Treatment"]
        laymen_terms = disease_info["Laymen Terms"]
        
        symptoms_vec = symptoms_tfidf.astype(np.float32)
        _, indices = faiss_index.search(symptoms_vec, 1)
        similar_disease = df.iloc[indices[0][0]]["Disease"]
        similar_disease_name = label_encoder.inverse_transform([similar_disease])[0]
        
        return disease_name, treatment, laymen_terms, similar_disease_name, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None, None, None

# Streamlit UI
def main():
    st.set_page_config(page_title="Disease Diagnosis System", page_icon="🏥", layout="wide")
    
    # Custom CSS for better visuals
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Disease Diagnosis System")
    st.write("Enter or select symptoms to get a predicted disease, treatment, and more!")
    
    # Load model and artifacts
    model, tfidf, label_encoder, faiss_index, df, symptom_list = load_model_and_artifacts(model_folder)
    if model is None:
        st.error("Failed to load model. Please check the folder path and files.")
        return
    
    # Sidebar for settings and info
    with st.sidebar:
        st.header("Settings & Info")
        st.write("Adjust options or learn more about the app.")
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 70, help="Minimum confidence for displaying results")
        st.write(f"Model loaded from: {model_folder}")
        st.write("Built with a neural network and FAISS for symptom similarity.")
    
    # Tabs for input methods
    tab1, tab2 = st.tabs(["Manual Entry", "Symptom Checklist"])
    
    # Tab 1: Manual Entry
    with tab1:
        st.subheader("Manual Symptom Entry")
        symptoms_input = st.text_area("Enter symptoms (comma-separated, e.g., 'fever, cough, fatigue')", "")
        st.write("Or use the dropdown to add common symptoms:")
        selected_symptom = st.selectbox("Add a symptom", [""] + symptom_list, index=0)
        if st.button("Add Symptom") and selected_symptom:
            symptoms_input = f"{symptoms_input}, {selected_symptom}" if symptoms_input else selected_symptom
    
    # Tab 2: Symptom Checklist
    with tab2:
        st.subheader("Symptom Checklist")
        selected_symptoms = st.multiselect("Select your symptoms", symptom_list, help="Choose multiple symptoms")
        symptoms_input_checklist = ", ".join(selected_symptoms) if selected_symptoms else ""
    
    # Use the appropriate input based on tab activity
    symptoms = symptoms_input if tab1 else symptoms_input_checklist
    
    # Diagnose button and prediction logic
    if st.button("Diagnose", key="diagnose_btn"):
        if symptoms.strip():
            with st.spinner("Analyzing symptoms..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulate processing time
                    progress_bar.progress(i + 1)
                
                disease, treatment, laymen, similar_disease, confidence = predict(symptoms, model, tfidf, label_encoder, faiss_index, df)
                if disease and confidence >= confidence_threshold:
                    st.success("Diagnosis complete!")
                    
                    # Display results in expandable sections
                    with st.expander("Diagnosis Results", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Disease", disease)
                            st.write(f"**Confidence:** {confidence:.2f}%")
                        with col2:
                            st.metric("Similar Disease (FAISS)", similar_disease)
                    
                    with st.expander("Treatment & Explanation"):
                        st.write(f"**Treatment:** {treatment}")
                        st.write(f"**Laymen Terms:** {laymen}")
                    
                    st.info("Note: This is an AI-generated suggestion. Please consult a doctor for a professional diagnosis.")
                elif confidence < confidence_threshold:
                    st.warning(f"Prediction confidence ({confidence:.2f}%) is below the threshold ({confidence_threshold}%). Results may be unreliable.")
                else:
                    st.error("Prediction failed. Please try again.")
        else:
            st.warning("Please enter or select some symptoms to diagnose.")
    
    # Footer
    st.markdown("---")
    st.write("© 2023 Disease Diagnosis System | Project By OMDENA KADUNA IMPACT HUB 2025")

if __name__ == "__main__":
    main()
