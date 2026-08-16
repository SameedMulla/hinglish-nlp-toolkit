# Code-Mixed Language Understanding (Hinglish NLP Toolkit)

A unified multitask NLP framework designed for **Code-Mixed Hindi-English (Hinglish)** text. 
This project provides a single, unified prediction interface (`UnifiedPredictor`) and an interactive **Streamlit** web application to test inference across 5 distinct NLP tasks.

## Supported Tasks
1. **Bidirectional Translation**: Code-Mixed ↔ English (Powered by HuggingFace `Helsinki-NLP/opus-mt-hi-en` and `opus-mt-en-hi`)
2. **Sentiment Analysis**: Multi-class classification (Positive, Negative, Neutral)
3. **Named Entity Recognition (NER)**: BIO token classification for PER, LOC, ORG, MISC.
4. **Question Answering (QA)**: Extractive reading comprehension.
5. **Intent Detection**: Multi-class conversational intent classification.

---

## 🚀 Getting Started

### 1. Clone & Install Dependencies
Ensure you have Python 3.8+ installed.

```bash
git clone https://github.com/SameedMulla/hinglish-nlp-toolkit.git
cd hinglish-nlp-toolkit

# Optional: Create a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Setup Datasets (COMI-LINGUA)
This toolkit utilizes the **COMI-LINGUA** datasets for training the baselines.
1. Place the COMI-LINGUA dataset CSV files (`MT_train.csv`, `NER_train.csv`, etc.) into the `data/COMI-LINGUA/` directory.
2. Run the preprocessing script to clean and normalize the data into JSON-lines format:
```bash
python data/scripts/preprocess.py --tasks all
```
*(This will generate task-ready files inside `data/processed/`)*

### 3. Train Baseline Models
The project relies on classical machine learning baselines (TF-IDF + SVM, and CRF Sequence Taggers) as fast, lower-bound models for classification and NER tasks.
Run the training script to generate the model weights:
```bash
python scripts/train_baselines.py
```
*(Models will be saved in the `models/` directory)*

### 4. Run the Web Application
Test all the models interactively through the built-in Streamlit interface. The app handles script detection, dynamic model loading, and bidirectional translation.

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Project Architecture

* `app/`: Streamlit web interface (`streamlit_app.py`).
* `config/`: YAML files for global hyperparameters and task-specific configurations.
* `data/`: Dataset storage (`raw/`, `processed/`, `COMI-LINGUA/`).
* `models/`: Checkpoints and serialized `.joblib` / `.model` files.
* `src/`: 
  * `inference/predictor.py`: The `UnifiedPredictor` API handling dynamic model loading and fallback heuristics.
  * `preprocessing/`: Normalization and data cleaning routines.
  * `models/`: Baseline model architectures (`baselines.py`).
* `scripts/`: CLI tools for training, evaluating, and testing.

## Model Fallback Strategy
To ensure maximum stability in production or demos, the `UnifiedPredictor` uses a graceful fallback strategy:
1. **Transformer Checkpoint**: Attempts to load fine-tuned transformer weights from `models/{task}/`.
2. **Baseline ML Model**: Falls back to the serialized TF-IDF/CRF models trained via `train_baselines.py`.
3. **Rule-Based Heuristic**: If no models are available, utilizes safe, hard-coded keyword heuristic logic to provide an answer.
