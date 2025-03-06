# Movie Review Sentiment Analysis App

## Project Overview
This project is a **Movie Review Sentiment Analysis App** built with **Streamlit**. It uses a **LoRA fine-tuned DistilBERT model** for sentiment classification, enabling efficient and accurate predictions.

## Features
- **Movie Review Sentiment Analysis** (Positive, Negative, Neutral)
- **LoRA-based Fine-Tuned Model** for efficient inference
- **Streamlit UI** for user-friendly interaction
- **Pre-trained Transformers** (Hugging Face) for text processing
- **Pickle-based Model Loading**

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Create a Virtual Environment
```bash
python -m venv env  # Windows
source env/bin/activate  # Mac/Linux
env\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Running the Streamlit App
```bash
streamlit run app.py
```

### 2. Using the App
- **Load the Model**: Provide the path to the `.pkl` model file.
- **Enter a Movie Review**: Type or paste a review to analyze.
- **Use Example Reviews**: Click predefined positive, negative, or neutral movie reviews.
- **Get Sentiment Analysis**: View predictions with confidence scores and sentiment visualization.

## Model Training (Optional)
To fine-tune the model using LoRA, you can train it using:
```bash
python src/train.py
```

## Prediction via Command Line (Optional)
If you want to use the model without the UI:
```bash
python src/predict.py --text "This movie was fantastic!"
```

## Dependencies
- Python 3.8+
- Streamlit
- Transformers (Hugging Face)
- PyTorch
- peft (LoRA Fine-Tuning)

## Contributing
Feel free to open issues and submit pull requests.

## License
MIT License


