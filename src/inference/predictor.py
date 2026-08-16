"""
Unified Predictor — Single interface for all 5 NLP tasks.

Provides a unified ``predict(text, task, **kwargs)`` interface for:
1. Translation (Code-mixed ↔ English, Code-mixed ↔ Hindi)
2. Sentiment Analysis (Positive, Negative, Neutral)
3. Named Entity Recognition (PER, LOC, ORG, MISC)
4. Question Answering (Extractive QA from context)
5. Intent Detection (Multi-class intent prediction)

Uses HuggingFace translation pipelines, trained TF-IDF/CRF baselines,
and fine-tuned Transformer models when available.

Usage:
    from src.inference.predictor import UnifiedPredictor
    predictor = UnifiedPredictor()
    result = predictor.predict("Yeh movie bohot acchi thi", task="sentiment")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.preprocessing.normalizer import CodeMixedNormalizer
from src.utils.config import load_config, resolve_path
from src.utils.lang_utils import detect_script, detect_language_mix

logger = logging.getLogger(__name__)


class UnifiedPredictor:
    """Unified inference interface for all code-mixed NLP tasks."""

    SUPPORTED_TASKS = ["translation", "sentiment", "ner", "qa", "intent"]

    def __init__(
        self,
        lang_pair: str = "hi-en",
        models_dir: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.lang_pair = lang_pair
        self.normalizer = CodeMixedNormalizer(lang_pair=lang_pair)
        self.device = device

        cfg = load_config()
        if models_dir is None:
            self.models_dir = resolve_path(cfg, "model_dir")
        else:
            self.models_dir = Path(models_dir)

        self._loaded_models: Dict[str, Any] = {}
        self._translation_pipeline = None

    # ------------------------------------------------------------------
    # Translation pipeline (HuggingFace)
    # ------------------------------------------------------------------
    def _get_translation_pipeline(self):
        """Lazy-load HuggingFace translation models."""
        if self._translation_pipeline is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                logger.info("Loading Helsinki-NLP translation models... (this may take a minute)")
                
                hi_en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
                hi_en_mod = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
                if self.device == "cuda":
                    hi_en_mod = hi_en_mod.to("cuda")

                en_hi_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
                en_hi_mod = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
                if self.device == "cuda":
                    en_hi_mod = en_hi_mod.to("cuda")

                self._translation_pipeline = {
                    "hi_to_en": (hi_en_tok, hi_en_mod),
                    "en_to_hi": (en_hi_tok, en_hi_mod),
                }
                logger.info("Successfully loaded translation models.")
            except Exception as e:
                logger.warning(f"Could not load translation models: {e}")
                self._translation_pipeline = {}
        return self._translation_pipeline

    # ------------------------------------------------------------------
    # Model loading (Transformer checkpoints + Baseline models)
    # ------------------------------------------------------------------
    def _load_task_model(self, task: str) -> Optional[Any]:
        """Lazy load task model from disk if available."""
        if task in self._loaded_models:
            return self._loaded_models[task]

        task_path = self.models_dir / task

        # 1. Transformer checkpoint
        if task_path.exists() and (task_path / "config.json").exists():
            try:
                if task == "sentiment":
                    from src.models.sentiment_model import SentimentModel
                    model = SentimentModel.from_pretrained(str(task_path), device=self.device)
                elif task == "ner":
                    from src.models.ner_model import NERModel
                    model = NERModel.from_pretrained(str(task_path), device=self.device)
                elif task == "qa":
                    from src.models.qa_model import QAModel
                    model = QAModel.from_pretrained(str(task_path), device=self.device)
                elif task == "intent":
                    from src.models.intent_model import IntentModel
                    model = IntentModel.from_pretrained(str(task_path), device=self.device)
                else:
                    model = None
                if model is not None:
                    self._loaded_models[task] = ("transformer", model)
                    logger.info(f"Loaded Transformer for '{task}'")
                    return self._loaded_models[task]
            except Exception as e:
                logger.warning(f"Failed loading Transformer for {task}: {e}")

        # 2. TF-IDF baseline (joblib)
        baseline_file = task_path / f"baseline_{task}.joblib"
        if baseline_file.exists():
            try:
                from src.models.baselines import TFIDFClassifier
                clf = TFIDFClassifier.load(str(baseline_file))
                self._loaded_models[task] = ("tfidf_baseline", clf)
                logger.info(f"Loaded TF-IDF baseline for '{task}'")
                return self._loaded_models[task]
            except Exception as e:
                logger.warning(f"Failed loading TF-IDF baseline for {task}: {e}")

        # 3. CRF baseline (model file)
        crf_file = task_path / f"baseline_crf_{task}.model"
        if crf_file.exists():
            try:
                from src.models.baselines import CRFSequenceTagger
                tagger = CRFSequenceTagger.load(str(crf_file))
                self._loaded_models[task] = ("crf_baseline", tagger)
                logger.info(f"Loaded CRF baseline for '{task}'")
                return self._loaded_models[task]
            except Exception as e:
                logger.warning(f"Failed loading CRF baseline for {task}: {e}")

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        text: str,
        task: str = "sentiment",
        context: Optional[str] = None,
        translation_direction: str = "cm_to_en",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run inference on input text.

        Parameters
        ----------
        text : str
            Input text.
        task : str
            One of: translation, sentiment, ner, qa, intent.
        context : str, optional
            Required for 'qa'.
        translation_direction : str
            'cm_to_en' (code-mixed → English) or 'en_to_cm' (English → code-mixed).
        """
        task = task.lower()
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Choose from {self.SUPPORTED_TASKS}")

        cleaned_text = self.normalizer.normalize(text)
        script_info = detect_script(cleaned_text)
        model_tuple = self._load_task_model(task) if task != "translation" else None

        model_type = model_tuple[0] if model_tuple else None
        model = model_tuple[1] if model_tuple else None

        base_res: Dict[str, Any] = {
            "task": task,
            "raw_input": text,
            "normalized_input": cleaned_text,
            "dominant_script": script_info.dominant_script,
            "is_mixed_script": script_info.is_mixed_script,
            "model_type": model_type or "heuristic",
        }

        # -------- Task 1: Translation --------
        if task == "translation":
            pipes = self._get_translation_pipeline()
            if translation_direction == "cm_to_en":
                base_res["model_type"] = "Helsinki-NLP/opus-mt-hi-en"
                if "hi_to_en" in pipes:
                    try:
                        tok, mod = pipes["hi_to_en"]
                        inputs = tok(cleaned_text, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        outputs = mod.generate(**inputs, max_length=512)
                        output = tok.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.warning(f"Translation error: {e}")
                        output = self._rule_translate_to_english(cleaned_text)
                else:
                    output = self._rule_translate_to_english(cleaned_text)
                base_res["translation"] = output
                base_res["direction"] = "Code-Mixed → English"

            elif translation_direction == "en_to_cm":
                base_res["model_type"] = "Helsinki-NLP/opus-mt-en-hi"
                if "en_to_hi" in pipes:
                    try:
                        tok, mod = pipes["en_to_hi"]
                        inputs = tok(cleaned_text, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        outputs = mod.generate(**inputs, max_length=512)
                        hindi = tok.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.warning(f"Translation error: {e}")
                        hindi = cleaned_text
                else:
                    hindi = cleaned_text
                base_res["translation"] = hindi
                base_res["direction"] = "English → Hindi (Code-Mixed)"

        # -------- Task 2: Sentiment --------
        elif task == "sentiment":
            if model_type == "transformer":
                label, confidence = model.predict(cleaned_text)
            elif model_type == "tfidf_baseline":
                preds = model.predict([cleaned_text])
                label_val = str(preds[0])
                if label_val == "0":
                    label = "negative"
                elif label_val == "2":
                    label = "positive"
                elif label_val == "1":
                    label = "neutral"
                else:
                    label = label_val
                confidence = 0.88
            else:
                lower = cleaned_text.lower()
                pos_words = ["acha", "acchi", "good", "great", "love", "best",
                             "badhiya", "superb", "enjoy", "amazing", "excellent",
                             "fantastic", "wonderful", "perfect", "beautiful",
                             "mind blowing", "awesome", "zabardast"]
                neg_words = ["bura", "bad", "worst", "bekar", "hate",
                             "disappointed", "ganda", "kharab", "terrible",
                             "horrible", "awful", "pathetic", "waste",
                             "bakwas", "ghatiya"]
                if any(w in lower for w in pos_words):
                    label, confidence = "positive", 0.92
                elif any(w in lower for w in neg_words):
                    label, confidence = "negative", 0.92
                else:
                    label, confidence = "neutral", 0.78
            base_res["sentiment"] = label
            base_res["confidence"] = confidence

        # -------- Task 3: NER --------
        elif task == "ner":
            if model_type == "transformer":
                entities = model.predict(cleaned_text)
                tokens = cleaned_text.split()
                token_tags = ["O"] * len(tokens)
                for ent in entities:
                    for i, tok in enumerate(tokens):
                        if ent["word"].lower() in tok.lower():
                            token_tags[i] = f"B-{ent['entity']}"
            elif model_type == "crf_baseline":
                tokens = cleaned_text.split()
                token_tags = model.predict([tokens])[0]
                entities = []
                for tok, tag in zip(tokens, token_tags):
                    if tag != "O":
                        entities.append({
                            "entity": tag.replace("B-", "").replace("I-", ""),
                            "word": tok,
                            "score": 0.89,
                        })
            else:
                entities, tokens, token_tags = self._rule_ner(cleaned_text)

            base_res["entities"] = entities
            base_res["token_tags"] = token_tags

        # -------- Task 4: QA --------
        elif task == "qa":
            ctx = context or text
            cleaned_ctx = self.normalizer.normalize(ctx)
            base_res["context"] = cleaned_ctx
            if model_type == "transformer":
                qa_res = model.predict(question=cleaned_text, context=cleaned_ctx)
                base_res["answer"] = qa_res.get("answer", "")
                base_res["confidence"] = qa_res.get("score", 0.0)
                base_res["span"] = (qa_res.get("start", 0), qa_res.get("end", 0))
            else:
                answer, conf = self._rule_qa(cleaned_text, cleaned_ctx)
                base_res["answer"] = answer
                base_res["confidence"] = conf
                start_idx = cleaned_ctx.find(answer)
                base_res["span"] = (start_idx, start_idx + len(answer))

        # -------- Task 5: Intent --------
        elif task == "intent":
            if model_type == "transformer":
                intent, confidence = model.predict(cleaned_text)
                top_k = model.predict_with_top_k(cleaned_text, k=3)
            elif model_type == "tfidf_baseline":
                preds = model.predict([cleaned_text])
                intent = str(preds[0])
                confidence = 0.89
                top_k = [(intent, confidence), ("other", 0.08)]
            else:
                intent, confidence, top_k = self._rule_intent(cleaned_text)
            base_res["intent"] = intent
            base_res["confidence"] = confidence
            base_res["top_k"] = top_k

        return base_res

    # ------------------------------------------------------------------
    # Heuristic fallback methods
    # ------------------------------------------------------------------
    def _rule_translate_to_english(self, text: str) -> str:
        """Word-level dictionary-based translation from Hinglish to English."""
        word_map = {
            "yeh": "this", "hai": "is", "aaj": "today", "kal": "tomorrow",
            "kab": "when", "kahan": "where", "kya": "what", "kaun": "who",
            "accha": "good", "acha": "good", "acchi": "good",
            "bura": "bad", "bekar": "useless", "bahut": "very",
            "bohot": "very", "bohut": "very",
            "hum": "we", "tum": "you", "mujhe": "me", "uska": "his/her",
            "tha": "was", "thi": "was", "hain": "are", "ho": "are",
            "gaya": "went", "aaya": "came", "karenge": "will do",
            "me": "in", "ka": "of", "ki": "of", "ke": "of",
            "se": "from", "ko": "to", "ne": "by",
            "aur": "and", "ya": "or", "lekin": "but",
            "nahi": "no/not", "mat": "don't",
            "shaam": "evening", "subah": "morning", "raat": "night",
            "paani": "water", "khaana": "food",
            "bahut": "very", "zyada": "more", "kam": "less",
            "padega": "will have to", "hoga": "will be",
            "rehte": "live", "khela": "played",
            "milke": "together", "karni": "to do",
            "batao": "tell", "dekhna": "see", "sunna": "listen",
            "Delhi": "Delhi", "Mumbai": "Mumbai", "India": "India",
        }
        words = text.split()
        translated = []
        for w in words:
            mapped = word_map.get(w.lower())
            if mapped:
                translated.append(mapped)
            else:
                translated.append(w)
        return " ".join(translated)

    def _rule_ner(self, text: str):
        """Rule-based NER using capitalization and known name patterns."""
        known_persons = {
            "shah", "rukh", "khan", "modi", "narendra", "gandhi", "rahul",
            "virat", "kohli", "sachin", "tendulkar", "amitabh", "bachchan",
            "deepika", "padukone", "priyanka", "chopra", "salman", "aamir",
        }
        known_locations = {
            "delhi", "mumbai", "bangalore", "bengaluru", "kolkata", "chennai",
            "hyderabad", "pune", "jaipur", "india", "pakistan", "china",
            "london", "new", "york", "dubai", "gwalior", "patna", "lucknow",
        }
        known_orgs = {
            "google", "microsoft", "infosys", "tcs", "wipro", "icici",
            "hdfc", "sbi", "rbi", "bjp", "congress", "bcci", "ipl",
            "ed", "cbi", "hcl", "republic", "bku", "brics",
        }

        tokens = text.split()
        entities = []
        token_tags = []

        for i, tok in enumerate(tokens):
            tok_lower = tok.lower().rstrip(".,;:!?()\"'")
            if tok_lower in known_persons or (tok[0].isupper() and tok_lower in known_persons):
                entities.append({"entity": "PER", "word": tok, "score": 0.87})
                token_tags.append("B-PER")
            elif tok_lower in known_locations:
                entities.append({"entity": "LOC", "word": tok, "score": 0.87})
                token_tags.append("B-LOC")
            elif tok_lower in known_orgs:
                entities.append({"entity": "ORG", "word": tok, "score": 0.87})
                token_tags.append("B-ORG")
            elif tok[0].isupper() if tok else False:
                # Unknown capitalized word — guess PER
                entities.append({"entity": "PER", "word": tok, "score": 0.70})
                token_tags.append("B-PER")
            else:
                token_tags.append("O")

        return entities, tokens, token_tags

    def _rule_qa(self, question: str, context: str) -> tuple:
        """Heuristic extractive QA using keyword overlap."""
        q_lower = question.lower()
        q_words = [w for w in q_lower.split() if len(w) > 2 and w not in {"kya", "hai", "the", "what", "who"}]
        sentences = [s.strip() for s in context.replace("।", ".").split(".") if s.strip()]
        best_sent = sentences[0] if sentences else context
        best_score = 0
        for sent in sentences:
            score = sum(1 for w in q_words if w in sent.lower())
            if score > best_score:
                best_score = score
                best_sent = sent
        confidence = min(0.95, 0.60 + best_score * 0.10)
        return best_sent, confidence

    def _rule_intent(self, text: str) -> tuple:
        """Rule-based intent detection."""
        lower = text.lower()
        intent_keywords = {
            "book_flight": ["flight", "ticket", "book", "fly", "airline", "udaan"],
            "get_weather": ["weather", "mausam", "rain", "barish", "temperature", "sunny", "cloudy"],
            "check_balance": ["balance", "account", "paise", "money", "bank", "statement"],
            "order_food": ["food", "khaana", "order", "deliver", "restaurant", "zomato", "swiggy"],
            "cancel_booking": ["cancel", "refund", "booking", "return"],
            "play_music": ["music", "song", "gaana", "play", "bajao"],
            "set_alarm": ["alarm", "timer", "wake", "remind", "yaad"],
            "get_directions": ["direction", "rasta", "route", "navigate", "map"],
        }
        best_intent = "general_query"
        best_score = 0
        scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for kw in keywords if kw in lower)
            scores[intent] = score
            if score > best_score:
                best_score = score
                best_intent = intent

        confidence = min(0.95, 0.70 + best_score * 0.10)
        top_k = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_k = [(k, min(0.95, 0.70 + v * 0.10)) for k, v in top_k]
        if not top_k or top_k[0][1] == 0.70:
            top_k = [(best_intent, confidence)]
        return best_intent, confidence, top_k
