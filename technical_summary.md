# Technical Summary: Deep Learning for Code-Mixed Hindi-English Natural Language Processing

## 1. Introduction
This project presents an end-to-end framework for analyzing code-mixed Hindi-English (Hinglish) text. Recognizing that code-mixing introduces unique vocabulary overlaps, non-standard spellings, and rapid script-switching, we have transitioned from legacy machine learning baselines (e.g., TF-IDF with SVMs, and Conditional Random Fields) to modern parameter-efficient transformer-based architectures.

Our framework features a single `UnifiedPredictor` API capable of serving predictions for 5 core tasks simultaneously.

## 2. Model Architecture and Backbone
Instead of monolingual models (like English-only BERT), we utilized **MuRIL** (`google/muril-base-cased` - Multilingual Representations for Indian Languages).
MuRIL is explicitly pre-trained on 17 Indian languages and English, along with transliterated versions of these languages. This makes it theoretically and empirically superior for Romanized Hindi (Hinglish) processing over standard `mBERT` or `XLM-R`.

For bidirectional translation (Hinglish $\leftrightarrow$ English), we utilized **Helsinki-NLP's Opus-MT** (`opus-mt-hi-en` and `opus-mt-en-hi`). Since Romanized Hindi shares phonetic traits mapped natively into these seq2seq architectures, zero-shot and lightweight inference yields functional semantic translation.

## 3. Implementation of Fine-Tuning
The transition to deep learning involved setting up supervised fine-tuning loops for each specific downstream task using PyTorch and the HuggingFace `Trainer` API. 

### 3.1 Task Formulation
- **Sentiment Analysis**: Formulated as Sequence Classification. The final hidden state of the `[CLS]` token is projected into a 3-dimensional space (Positive, Negative, Neutral).
- **Intent Detection**: Formulated similarly to Sentiment Analysis, but with a dense classification head sized to the number of distinct intents ($N=26$) present in the synthetic conversational dataset.
- **Named Entity Recognition (NER)**: Formulated as Token Classification. The hidden state corresponding to each token in the input sequence is mapped to BIO-tagged entities (PER, LOC, ORG, MISC, O).

### 3.2 Hyperparameter Configuration
To ensure stable gradients across all tasks, we standardized the following fine-tuning configurations:
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: $2 \times 10^{-5}$ to $3 \times 10^{-5}$
- **Warmup Ratio**: 10% of total training steps
- **Batch Size**: 32 for sequence classification, 16 for token classification
- **Evaluation**: Macro F1-score across validation epochs

## 4. Inference Engine (UnifiedPredictor)
A significant contribution of this framework is the inference engine's robust deployment capability. The `UnifiedPredictor` dynamically instantiates the correct architecture based on task heuristics.

**Fallback Strategy for Reliability:**
1. **Primary**: Load the fine-tuned Transformer model from `models/{task}/`.
2. **Secondary**: If deep learning weights are missing (or incompatible due to hardware constraints), gracefully fallback to the classical baselines (e.g., `crf_baseline`, `tfidf_baseline`).
3. **Tertiary**: Rule-based keyword matching heuristic.

## 5. Conclusion
By migrating to `google/muril-base-cased` and employing standard HuggingFace tokenization, we have successfully modernized the code-mixed NLP pipeline. The framework is heavily modular, allowing the dynamic swapping of the backbone architecture via simple YAML configuration changes. Future research can easily build upon this by experimenting with PEFT (Parameter-Efficient Fine-Tuning) such as LoRA on larger foundation models, or extending the training epochs for benchmark-level metrics.
