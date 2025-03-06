import streamlit as st
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the pickled model
@st.cache_resource
def load_model(model_path):
    """Load the pickled model and tokenizer"""
    try:
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        if isinstance(bundle, dict):
            model = bundle.get('model')
            tokenizer = bundle.get('tokenizer')
            if not tokenizer and 'model_checkpoint' in bundle:
                # If tokenizer not in pickle, load from the checkpoint name
                tokenizer = AutoTokenizer.from_pretrained(bundle['model_checkpoint'])
        else:
            # If the pickle is just the model
            model = bundle
            # Default to DistilBERT tokenizer if not specified
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to make predictions
def predict_sentiment(text, model, tokenizer):
    """Predict sentiment from input text"""
    # Prepare the model for inference
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    # Get prediction class (0: Negative, 1: Positive)
    prediction = torch.argmax(probs, dim=1).item()
    
    # Map to label
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    # Get confidence scores
    neg_prob = probs[0][0].item()
    pos_prob = probs[0][1].item()
    
    return sentiment, neg_prob, pos_prob

# Sidebar for app settings
st.sidebar.title("Model Settings")
model_path = st.sidebar.text_input(
    "Path to model file (.pkl)", 
    value="model_bundle.pkl",
    help="Enter the path to your pickled model file"
)

# Main app layout
st.title("💬 Sentiment Analysis App")
st.write("Upload your pickled model and analyze text sentiment")

# Load model button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model... This might take a minute."):
        model, tokenizer = load_model(model_path)
        if model and tokenizer:
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model. Check the file path.")

# Input area
text_input = st.text_area(
    "Enter a movie review to analyze:",
    height=150,
    value=st.session_state.get('text_input', ""),
    placeholder="Type or paste a movie review here to analyze its sentiment..."
)

# Example buttons
st.markdown("### Or try an example movie review:")
col1, col2, col3 = st.columns(3)

if col1.button("Positive Review"):
    st.session_state['text_input'] = "This movie was absolutely amazing! The storyline was engaging, the acting was top-notch, and the cinematography was breathtaking. A must-watch!"

if col2.button("Negative Review"):
    st.session_state['text_input'] = "This was one of the worst movies I've seen. The plot made no sense, the acting was wooden, and I was bored the entire time. Total waste of time."

if col3.button("Neutral Review"):
    st.session_state['text_input'] = "The movie had some interesting moments, but overall, it was just okay. The acting was decent, but the pacing felt a bit slow."


# Update text area if example was clicked
if 'text_input' in st.session_state:
    text_input = st.session_state['text_input']

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if not text_input:
        st.warning("Please enter some text to analyze")
    elif 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.warning("Please load the model first")
    else:
        with st.spinner("Analyzing..."):
            sentiment, neg_prob, pos_prob = predict_sentiment(
                text_input, 
                st.session_state['model'], 
                st.session_state['tokenizer']
            )
            
            # Display results
            st.markdown("## Results")
            
            # Sentiment result with appropriate styling
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment} 😊")
            else:
                st.error(f"Sentiment: {sentiment} 😞")
            
            # Confidence visualization
            st.markdown("### Confidence Scores")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Negative", f"{neg_prob:.2%}")
            with cols[1]:
                st.metric("Positive", f"{pos_prob:.2%}")
            
            # Progress bars
            st.markdown("### Sentiment Distribution")
            st.progress(pos_prob, text="Positive")
            st.progress(neg_prob, text="Negative")
            
            # Analysis details
            st.markdown("### Text Analysis")
            st.info(f"Analyzed {len(text_input)} characters ({len(text_input.split())} words)")

# Footer
st.markdown("---")
st.markdown("Sentiment Analysis App - Built with Streamlit and PyTorch")
