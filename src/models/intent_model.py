"""
Intent detection model — multi-class classification on code-mixed text.

Architecturally identical to the sentiment model but with a different
label set. Separated for clarity and independent configuration.

Usage:
    from src.models.intent_model import IntentModel
    model = IntentModel(backbone="google/muril-base-cased", num_labels=26)
    model.train(train_dataset, val_dataset)
    intent, score = model.predict("mujhe Delhi ka flight book karna hai")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class IntentModel:
    """Transformer-based intent classifier.

    Parameters
    ----------
    backbone : str
        Hugging Face model name.
    num_labels : int
        Number of intent classes (determined at runtime from dataset).
    label_map : dict, optional
        Mapping from label ID to intent name.
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_length : int
        Maximum sequence length.
    """

    def __init__(
        self,
        backbone: str = "google/muril-base-cased",
        num_labels: int = 26,
        label_map: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
        max_length: int = 128,  # Intents are typically short utterances
    ) -> None:
        self.backbone = backbone
        self.num_labels = num_labels
        self.label_map = label_map or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        logger.info(f"Loading intent model: {backbone} ({num_labels} labels)")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=num_labels
        )
        self.model.to(self.device)

    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenize texts for classification."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def _compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Compute accuracy and macro-F1."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "models/intent",
        num_epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        **kwargs: Any,
    ) -> None:
        """Fine-tune the model for intent detection."""
        # Build label map from data if not provided
        if not self.label_map and "intent" in train_dataset.column_names:
            unique_labels = sorted(set(train_dataset["intent"]))
            self.label_map = {i: str(l) for i, l in enumerate(unique_labels)}
            label2id = {str(l): i for i, l in enumerate(unique_labels)}
            self.num_labels = len(unique_labels)

            # Remap string labels to integer IDs
            train_dataset = train_dataset.map(
                lambda x: {"label": label2id[str(x["intent"])]},
            )
            if val_dataset is not None:
                val_dataset = val_dataset.map(
                    lambda x: {"label": label2id[str(x["intent"])]},
                )

        # Tokenize
        cols_to_remove = [c for c in train_dataset.column_names if c not in ("label",)]
        train_tokenized = train_dataset.map(
            self._tokenize_function, batched=True, remove_columns=cols_to_remove,
        )
        val_tokenized = None
        if val_dataset is not None:
            cols_to_remove_val = [c for c in val_dataset.column_names if c not in ("label",)]
            val_tokenized = val_dataset.map(
                self._tokenize_function, batched=True, remove_columns=cols_to_remove_val,
            )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch" if val_tokenized else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if val_tokenized else False,
            metric_for_best_model="macro_f1",
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to="none",
            **kwargs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting intent model training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single utterance.

        Returns
        -------
        tuple of (str, float)
            Predicted intent name and confidence score.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        intent_name = self.label_map.get(pred_id, str(pred_id))
        return intent_name, round(confidence, 4)

    def predict_with_top_k(
        self, text: str, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k intent predictions with scores."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        top_k = torch.topk(probs, k=min(k, self.num_labels))
        results = []
        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            intent_name = self.label_map.get(idx, str(idx))
            results.append((intent_name, round(score, 4)))
        return results

    @classmethod
    def from_pretrained(
        cls, path: str, num_labels: int = 26, **kwargs: Any
    ) -> "IntentModel":
        """Load a fine-tuned intent model from disk."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        instance.num_labels = num_labels
        instance.label_map = kwargs.get("label_map", {})
        instance.max_length = kwargs.get("max_length", 128)
        instance.backbone = path
        return instance
