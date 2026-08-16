# Code-Mixed Language Understanding (Hindi-English NLP Multitask Toolkit)

A comprehensive, modular NLP toolkit for **Hindi-English (Hinglish) code-mixed text**, designed for research and deployment across **5 core tasks**:

1. **Translation**: Code-mixed Hindi-English → Hindi translation (`mT5`, `IndicTrans2`)
2. **Sentiment Analysis**: Multi-class sentiment classification (`mBERT`, `XLM-R`, `MuRIL`)
3. **Named Entity Recognition (NER)**: BIO token-level classification (`mBERT`, `XLM-R`, `MuRIL`)
4. **Question Answering (QA)**: Extractive reading comprehension (`MuRIL`)
5. **Intent Detection**: Multi-class intent classification (`MuRIL`, synthetic code-mixing)

Designed from the ground up for **future language pair extensibility** (e.g. Tamil-English, Bengali-English) via YAML configs and parameterised preprocessing.

---

## 📁 Repository Structure

```
c:\Projects\NLP\
├── README.md                          # Main project documentation
├── requirements.txt                   # Dependency manifest
├── config/                            # Modular YAML configurations
│   ├── base.yaml                      # Global hyperparameters & paths
│   ├── translation.yaml               # Task override for translation
│   ├── sentiment.yaml                 # Task override for sentiment
│   ├── ner.yaml                       # Task override for NER
│   ├── qa.yaml                        # Task override for QA
│   └── intent.yaml                    # Task override for intent
├── data/
│   ├── raw/                           # Raw downloaded datasets
│   ├── processed/                     # Cleaned, split JSONL datasets
│   └── scripts/
│       ├── download_datasets.py       # Auto-downloader for HF & fallbacks
│       └── preprocess.py              # Unified pipeline (normalization & deduplication)
├── src/                               # Core Python package
│   ├── preprocessing/
│   │   ├── normalizer.py              # CodeMixedNormalizer (NFC, Romanized map, noise removal)
│   │   ├── tokenizer_utils.py         # Sub-word alignment for NER & sliding-window for QA
│   │   └── augmentation.py            # Back-translation & synthetic code-mixer
│   ├── models/
│   │   ├── baselines.py               # TF-IDF + Logistic Regression/SVM & CRF for NER
│   │   ├── translation_model.py       # Seq2Seq fine-tuning (mT5)
│   │   ├── sentiment_model.py         # Classification (mBERT/XLM-R/MuRIL)
│   │   ├── ner_model.py               # BIO token classification
│   │   ├── qa_model.py                # Extractive QA model
│   │   └── intent_model.py            # Multi-class intent model
│   ├── training/
│   │   └── trainer.py                 # Unified training orchestrator
│   ├── evaluation/
│   │   ├── metrics.py                 # BLEU, ROUGE, Macro-F1, EM, seqeval
│   │   └── error_analysis.py          # Code-mixed qualitative error categorisation
│   ├── inference/
│   │   └── predictor.py               # UnifiedPredictor across all 5 tasks
│   └── utils/
│       ├── config.py                  # Hierarchical YAML loader
│       └── lang_utils.py              # Script detection (Devanagari/Latin/Mixed)
├── app/
│   └── streamlit_app.py               # Interactive web app
├── notebooks/                         # Experimental & demonstration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_baseline_experiments.ipynb
│   ├── 04_transformer_experiments.ipynb
│   └── 05_evaluation_analysis.ipynb
├── scripts/                           # CLI entry points
│   ├── train.py                       # Training runner
│   ├── evaluate.py                    # Evaluation & error reporting runner
│   └── predict.py                     # Single-shot inference runner
└── tests/                             # Unit tests
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_evaluation.py
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Download Datasets
Download **COMI-LINGUA** and task-specific datasets from Hugging Face:
```bash
python data/scripts/download_datasets.py --tasks all
```

### 3. Preprocess Data
Run NFC normalization, Romanized Hindi spelling canonicalization, URL/noise stripping, and synthetic code-mixing:
```bash
python data/scripts/preprocess.py --tasks all --lang-pair hi-en
```

---

## ⚡ CLI Usage

### Train a Model
```bash
# Train Sentiment Analysis with MuRIL
python scripts/train.py --task sentiment --backbone google/muril-base-cased --epochs 10 --batch-size 32

# Train Translation with mT5
python scripts/train.py --task translation --backbone google/mt5-small --epochs 15
```

### Evaluate Model
```bash
python scripts/evaluate.py --task sentiment --save-report
```

### Run Inference
```bash
python scripts/predict.py --task sentiment --text "Yeh movie bohot achhi thi, direction mind blowing!"
python scripts/predict.py --task translation --text "Is problem ka solution fast nikalna padega"
python scripts/predict.py --task qa --text "Shah Rukh Khan kahan rehte hain?" --context "Shah Rukh Khan Mumbai me Mannat me rehte hain."
```

---

## 💻 Interactive Streamlit Demo Web App

Launch the interactive web application to test all 5 tasks visually:
```bash
streamlit run app/streamlit_app.py
```

Features:
- Real-time script & mix ratio inspection
- Visual entity highlights for NER
- Top-k candidate predictions for Intent Detection
- Language pair selector (`hi-en` Hindi-English + extensible `ta-en` Tamil-English template)

---

## 🧪 Unit Testing

Run the test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🌐 Extending to Future Languages

To add a new language pair (e.g. Tamil-English `ta-en`):
1. Add language specific romanization dictionary to `src/preprocessing/normalizer.py`.
2. Add new Unicode block ranges in `src/utils/lang_utils.py`.
3. Pass `--lang-pair ta-en` to preprocessing scripts. No model architecture or code changes required!
