# Import necessary libraries
import streamlit as st  # For building the web app
import torch  # PyTorch for model inference
import torch.nn as nn  # Neural network module from PyTorch
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to TF-IDF features
import faiss  # For efficient similarity search
import pickle  # For loading saved Python objects (e.g., model artifacts)
import os  # For file and directory operations
import plotly.express as px  # For interactive visualizations
import time  # For simulating processing delays

# Set Streamlit page configuration for layout and branding
st.set_page_config(
    page_title="Leveraging NLP in Medical Prescription",  # Title displayed in browser tab
    page_icon="🩺",  # Icon in browser tab (stethoscope emoji)
    layout="wide",  # Use full width of the screen
    initial_sidebar_state="expanded"  # Sidebar starts expanded
)

# Define device for PyTorch (use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model class (must match the structure used during training)
class OptimizedDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """Initialize the disease classifier model with a sequential network."""
        super(OptimizedDiseaseClassifier, self).__init__()
        # Define the neural network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),  # Input layer to hidden layer (512 units)
            nn.BatchNorm1d(512),  # Normalize outputs for stability
            nn.ReLU(),  # Activation function for non-linearity
            nn.Dropout(0.4),  # Dropout to prevent overfitting (40% drop rate)
            nn.Linear(512, 256),  # Hidden layer to smaller hidden layer (256 units)
            nn.BatchNorm1d(256),  # Normalize again
            nn.ReLU(),  # Another activation
            nn.Dropout(0.3),  # Dropout with 30% drop rate
            nn.Linear(256, num_classes)  # Output layer with number of disease classes
        )
        # Store dropout layers for dynamic adjustment
        self.dropout_layers = [layer for layer in self.network if isinstance(layer, nn.Dropout)]

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def update_dropout(self, p):
        """Dynamically update dropout probability for all dropout layers."""
        for layer in self.dropout_layers:
            layer.p = p

# Load model and artifacts with caching to avoid reloading on every run
@st.cache_resource
def load_model_and_artifacts(save_folder, num_classes):
    """Load the trained model, TF-IDF vectorizer, label encoder, and FAISS index."""
    # Load TF-IDF vectorizer to determine input dimension
    with open(os.path.join(save_folder, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    input_dim = len(tfidf.vocabulary_)

    # Initialize and load the model
    model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load trained weights
    model.eval()  # Set model to evaluation mode (no training)

    # Load label encoder for disease names
    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load FAISS index for similarity search
    faiss_index = faiss.read_index(os.path.join(save_folder, "faiss_index.bin"))

    return model, tfidf, label_encoder, faiss_index

# Load dataset and extract symptoms with caching for performance
@st.cache_data
def load_dataset_and_symptoms(data_path, _label_encoder):
    """Load the dataset and extract unique symptoms."""
    df = pd.read_csv(data_path)  # Load CSV file
    df.dropna(inplace=True)  # Remove rows with missing values
    df["Disease"] = _label_encoder.transform(df["Disease"])  # Encode disease labels
    all_symptoms = set()  # Use a set to collect unique symptoms
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])  # Split and clean symptoms
    return df, sorted(list(all_symptoms))  # Return DataFrame and sorted symptom list

# Prediction function with confidence scores and similarity search
def predict(symptoms, model, tfidf, label_encoder, faiss_index, df, confidence_threshold=0.5):
    """Predict disease, treatment, and similar conditions based on symptoms."""
    # Convert symptoms to TF-IDF vector
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    
    # Model inference
    with torch.no_grad():
        output = model(symptoms_tensor)  # Get raw model output
        probabilities = torch.softmax(output, dim=1)[0]  # Convert to probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get highest probability class
        confidence = probabilities[predicted_class].item()  # Confidence score
    
    # Decode predicted disease
    disease_name = label_encoder.inverse_transform([predicted_class])[0]

    # Fetch treatment from dataset if confidence is sufficient
    disease_rows = df[df["Disease"] == predicted_class]
    if disease_rows.empty or confidence < confidence_threshold:
        treatment = "Treatment information not available (low confidence or no match)"
    else:
        disease_info = disease_rows.iloc[0]
        treatment = disease_info.get("Treatment", "Treatment information not available")

    # Use FAISS to find the most similar disease based on symptoms
    symptoms_vec = symptoms_tfidf.astype(np.float32)
    _, indices = faiss_index.search(symptoms_vec, 1)  # Search for top 1 similar entry
    if indices[0][0] >= len(df):
        similar_disease_name = "Unknown"
    else:
        similar_disease = df.iloc[indices[0][0]]["Disease"]
        similar_disease_name = label_encoder.inverse_transform([similar_disease])[0]

    # Get top 3 predictions for visualization
    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_diseases = label_encoder.inverse_transform(top_indices.cpu().numpy())
    top_confidences = top_probs.cpu().numpy()

    return disease_name, treatment, similar_disease_name, confidence, top_diseases, top_confidences

# Main application function
def main():
    """Main function to run the Streamlit app."""
    # Define file paths (adjust these as needed based on your setup)
    save_folder = "model_classifier-002b"  # Directory containing model artifacts
    data_path = "processed_diseases-priority.csv"  # Dataset file

    # Load model and dataset
    try:
        num_classes = len(pd.read_csv(data_path)["Disease"].unique())  # Number of unique diseases
        model, tfidf, label_encoder, faiss_index = load_model_and_artifacts(save_folder, num_classes)
        df_filtered, common_symptoms = load_dataset_and_symptoms(data_path, label_encoder)
        st.success(f"Model and artifacts loaded successfully! Input dim: {len(tfidf.vocabulary_)}")
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return

    # Initialize session state for history and feedback
    if "history" not in st.session_state:
        st.session_state.history = []  # Store prediction history
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False  # Track feedback submission

    # Sidebar configuration
    st.sidebar.title("🩺 Disease Predictor")  # Sidebar title
    st.sidebar.markdown("Select or type symptoms to predict diseases.")  # Instruction
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053171.png", width=100)  # Icon

    # Advanced options in sidebar
    with st.sidebar.expander("Advanced Options", expanded=False):
        confidence_threshold = st.slider(
            "Confidence Threshold",  # Slider for setting confidence threshold
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence required for a prediction to be considered valid."
        )

    # Main content area
    st.title("Interactive Disease Prediction")  # App title
    st.markdown("**Explore symptoms and get real-time disease predictions!**")  # Subtitle

    # Symptom input section
    st.subheader("Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose common symptoms (or type below)",  # Dropdown for common symptoms
        options=common_symptoms,
        default=["Fever", "Cough"],  # Default selection
        help="Select from common symptoms or add custom ones below."
    )

    manual_symptoms = st.text_area(
        "Additional Symptoms (comma-separated)",  # Text area for manual input
        placeholder="e.g., fatigue, headache",
        height=100,
        help="Enter additional symptoms not listed above, separated by commas."
    )

    # Combine selected and manual symptoms
    symptoms = ", ".join(selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()])
    st.write(f"**Current Symptoms:** {symptoms if symptoms else 'None entered'}")  # Display symptoms

    # Validation message
    if symptoms:
        st.info("Symptoms valid! Click 'Predict' to proceed.")
    else:
        st.warning("Please enter or select at least one symptom.")

    # Predict button with progress simulation
    if st.button("Predict", key="predict_button"):
        if symptoms:
            progress_bar = st.progress(0)  # Show progress bar
            with st.spinner("Analyzing symptoms..."):  # Spinner during processing
                for i in range(100):
                    time.sleep(0.01)  # Simulate processing delay
                    progress_bar.progress(i + 1)
                try:
                    # Run prediction
                    disease, treatment, similar_disease, confidence, top_diseases, top_confidences = predict(
                        symptoms, model, tfidf, label_encoder, faiss_index, df_filtered, confidence_threshold
                    )

                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction")
                        # Color-coded confidence badge
                        badge_color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.5 else "red"
                        st.markdown(
                            f"**Disease:** {disease}  <span style='background-color:{badge_color};color:white;padding:2px 5px;border-radius:3px'>{confidence:.2%}</span>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Treatment:** {treatment}")

                    with col2:
                        st.subheader("Similar Condition")
                        st.warning(f"**Similar Disease:** {similar_disease}")
                        st.write("Consider this as a possible alternative.")

                    # Bar chart for top predictions
                    st.subheader("Top Predictions")
                    fig = px.bar(
                        x=top_diseases,
                        y=top_confidences,
                        labels={"x": "Disease", "y": "Confidence"},
                        title="Top 3 Predicted Diseases",
                        color=top_confidences,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Save to history
                    st.session_state.history.append({
                        "Symptoms": symptoms,
                        "Disease": disease,
                        "Treatment": treatment,
                        "Confidence": f"{confidence:.2%}",
                        "Similar Disease": similar_disease
                    })

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            progress_bar.empty()  # Clear progress bar
        else:
            st.error("No symptoms provided!")

    # Display prediction history
    if st.session_state.history:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)  # Show history as a table
        if st.button("Clear History"):
            st.session_state.history = []  # Reset history
            st.rerun()  # Refresh app

    # Feedback section
    st.subheader("Provide Feedback")
    st.write("Help us improve! Share your thoughts:")
    
    with st.expander("Quick Feedback", expanded=False):
        feedback_text = st.text_area("Your Feedback", placeholder="How was your experience? Any suggestions?")
        if st.button("Submit Feedback") and feedback_text:
            st.session_state.feedback_submitted = True  # Mark feedback as submitted
            st.success("Thank you for your feedback!")
        if st.session_state.feedback_submitted:
            st.write("Feedback submitted!")

    # Link to external feedback form
    google_form_link = "https://forms.gle/your-google-form-link"  # Replace with actual URL
    st.markdown(f"For detailed feedback, please fill out [this form]({google_form_link}).")

    # Footer with disclaimer and credits
    st.markdown("---")
    st.markdown(
        """
        **Note:** This app is for educational purposes only. Consult a healthcare professional for medical advice.
        Built with ❤️ using Streamlit, PyTorch, and Plotly
        Project initiated by Jamaludeen Madaki for Omdena Kaduna Impact Hub
        Collaboratively executed by the Project Team.
        """
    )

# Entry point of the script
if __name__ == "__main__":
    main()  # Run the app
