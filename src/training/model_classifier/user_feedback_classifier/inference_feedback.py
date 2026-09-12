#!/usr/bin/env python3
"""
Feedback Detector Inference

Compatible with: https://huggingface.co/llm-semantic-router/feedback-detector

Classifies the supplied follow-up text. Ambiguous feedback may require preceding
conversation context; confidence is a model score, not a correctness guarantee.
"""

# Multilingual examples retain their native scripts and punctuation.
# ruff: noqa: RUF001

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__:
    from .data_contract import ID2LABEL
else:
    from data_contract import ID2LABEL

DISPLAY_LIMIT = 45


@dataclass
class FeedbackResult:
    """Result from feedback classification."""

    label: str  # SAT, NEED_CLARIFICATION, WRONG_ANSWER, WANT_DIFFERENT
    confidence: float
    is_satisfied: bool
    all_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "is_satisfied": self.is_satisfied,
            "scores": self.all_scores,
        }

    def __repr__(self):
        return f"{self.label} ({self.confidence:.1%})"


class FeedbackDetector:
    """
    User feedback classifier for conversational AI.

    Detects user satisfaction from follow-up messages:
    - SAT: User is satisfied
    - NEED_CLARIFICATION: User needs more explanation
    - WRONG_ANSWER: System provided incorrect information
    - WANT_DIFFERENT: User wants alternative options
    """

    def __init__(
        self,
        model_path: str = "llm-semantic-router/feedback-detector",
        device: str | None = None,
        max_length: int = 512,
        revision: str | None = None,
    ):
        """
        Initialize the feedback detector.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading feedback detector from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, revision=revision
        )
        self.model.to(self.device)
        self.model.eval()

        labels = {int(key): value for key, value in self.model.config.id2label.items()}
        if labels != ID2LABEL:
            raise ValueError("Feedback checkpoint has an incompatible label mapping")
        capacity = self.model.config.max_position_embeddings
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or not 0 < max_length <= capacity
        ):
            raise ValueError(f"max_length must be between 1 and {capacity}")
        self.max_length = max_length
        self.model.config.reference_compile = False

        print(f"  Loaded on {self.device}")

    def classify(self, text: str) -> FeedbackResult:
        """
        Classify user feedback from follow-up message.

        Args:
            text: User's follow-up message, with available context if needed.

        Returns:
            FeedbackResult with label, confidence, and scores
        """
        inputs = self._encode(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Get all scores
        all_scores = {ID2LABEL[i]: probs[i].item() for i in range(len(ID2LABEL))}

        # Get prediction
        pred_idx = probs.argmax().item()
        label = ID2LABEL[pred_idx]
        confidence = probs[pred_idx].item()

        return FeedbackResult(
            label=label,
            confidence=confidence,
            is_satisfied=(label == "SAT"),
            all_scores=all_scores,
        )

    def classify_batch(self, texts: list[str]) -> list[FeedbackResult]:
        """Classify multiple follow-up messages."""
        if not texts:
            return []

        inputs = self._encode(texts)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(texts)):
            sample_probs = probs[i]
            all_scores = {
                ID2LABEL[j]: sample_probs[j].item() for j in range(len(ID2LABEL))
            }
            pred_idx = sample_probs.argmax().item()
            label = ID2LABEL[pred_idx]
            confidence = sample_probs[pred_idx].item()

            results.append(
                FeedbackResult(
                    label=label,
                    confidence=confidence,
                    is_satisfied=(label == "SAT"),
                    all_scores=all_scores,
                )
            )

        return results

    def _encode(self, texts: str | list[str]):
        """Reject over-budget requests without silently discarding feedback."""
        inputs = self.tokenizer(
            texts,
            truncation=False,
            padding=isinstance(texts, list),
            return_tensors="pt",
        )
        lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        over_budget = [
            (index, length)
            for index, length in enumerate(lengths)
            if length > self.max_length
        ]
        if over_budget:
            raise ValueError(
                f"Feedback input exceeds max_length={self.max_length}, including "
                f"special tokens: {over_budget}. Increase the explicit budget within "
                "the checkpoint capacity or shorten the input."
            )
        return inputs.to(self.device)

    def __call__(self, text: str) -> FeedbackResult:
        """Shortcut for classify()."""
        return self.classify(text)


def demo():
    """Demo the feedback detector."""
    print("=" * 70)
    print("FEEDBACK DETECTOR DEMO")
    print("=" * 70)

    # Try mmBERT model first, fall back to ModernBERT
    model_paths = [
        "models/mmbert_feedback_detector_merged",
        "models/mmbert_feedback_detector",
        "llm-semantic-router/feedback-detector",
    ]

    model_path = None
    for path in model_paths:
        if Path(path).exists() or "/" in path:
            model_path = path
            break

    detector = FeedbackDetector(model_path=model_path)

    # English test cases
    test_cases = [
        # SAT
        ("Thanks, that's helpful!", "SAT"),
        ("Perfect, exactly what I needed!", "SAT"),
        ("Great explanation!", "SAT"),
        # NEED_CLARIFICATION
        ("What do you mean by that?", "NEED_CLARIFICATION"),
        ("Can you explain more?", "NEED_CLARIFICATION"),
        ("I don't understand, could you elaborate?", "NEED_CLARIFICATION"),
        # WRONG_ANSWER
        ("No, that's wrong.", "WRONG_ANSWER"),
        ("That's incorrect, the answer is actually 42.", "WRONG_ANSWER"),
        ("You made a mistake there.", "WRONG_ANSWER"),
        # WANT_DIFFERENT
        ("Show me other options.", "WANT_DIFFERENT"),
        ("What else do you have?", "WANT_DIFFERENT"),
        ("I'd like to see alternatives.", "WANT_DIFFERENT"),
    ]

    print("\n🇺🇸 English Test Results:")
    print("-" * 70)

    correct = 0
    for text, expected in test_cases:
        result = detector.classify(text)
        match = "✅" if result.label == expected else "❌"
        if result.label == expected:
            correct += 1

        print(f'{match} "{text[:50]}"')
        print(f"   Expected: {expected}, Got: {result}")

    print(
        f"\nEnglish Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)"
    )

    # Multilingual test cases - mmBERT cross-lingual transfer!
    print("\n" + "=" * 70)
    print("🌍 MULTILINGUAL EXAMPLES (mmBERT cross-lingual transfer)")
    print("=" * 70)

    multilingual_cases = [
        # Spanish 🇪🇸
        ("¡Gracias, eso es muy útil!", "SAT", "Spanish"),
        ("No entiendo, ¿puedes explicar más?", "NEED_CLARIFICATION", "Spanish"),
        ("Eso está mal, la respuesta correcta es otra.", "WRONG_ANSWER", "Spanish"),
        ("Muéstrame otras opciones.", "WANT_DIFFERENT", "Spanish"),
        # French 🇫🇷
        ("Merci, c'est parfait!", "SAT", "French"),
        ("Je ne comprends pas, pouvez-vous expliquer?", "NEED_CLARIFICATION", "French"),
        ("Non, c'est faux.", "WRONG_ANSWER", "French"),
        ("Montrez-moi d'autres alternatives.", "WANT_DIFFERENT", "French"),
        # German 🇩🇪
        ("Danke, das ist sehr hilfreich!", "SAT", "German"),
        ("Was meinst du damit?", "NEED_CLARIFICATION", "German"),
        ("Das ist falsch.", "WRONG_ANSWER", "German"),
        ("Zeig mir andere Optionen.", "WANT_DIFFERENT", "German"),
        # Chinese 🇨🇳
        ("谢谢，这很有帮助！", "SAT", "Chinese"),
        ("我不明白，能解释一下吗？", "NEED_CLARIFICATION", "Chinese"),
        ("不对，答案是错的。", "WRONG_ANSWER", "Chinese"),
        ("给我看看其他选项。", "WANT_DIFFERENT", "Chinese"),
        # Japanese 🇯🇵
        ("ありがとう、とても助かりました！", "SAT", "Japanese"),
        ("どういう意味ですか？", "NEED_CLARIFICATION", "Japanese"),
        ("それは間違っています。", "WRONG_ANSWER", "Japanese"),
        ("他の選択肢を見せてください。", "WANT_DIFFERENT", "Japanese"),
        # Korean 🇰🇷
        ("감사합니다, 도움이 됐어요!", "SAT", "Korean"),
        ("무슨 말인지 모르겠어요.", "NEED_CLARIFICATION", "Korean"),
        ("그건 틀렸어요.", "WRONG_ANSWER", "Korean"),
        ("다른 옵션을 보여주세요.", "WANT_DIFFERENT", "Korean"),
        # Arabic 🇸🇦
        ("شكراً، هذا مفيد جداً!", "SAT", "Arabic"),
        ("لم أفهم، هل يمكنك التوضيح؟", "NEED_CLARIFICATION", "Arabic"),
        ("هذا خطأ.", "WRONG_ANSWER", "Arabic"),
        ("أرني خيارات أخرى.", "WANT_DIFFERENT", "Arabic"),
        # Portuguese 🇧🇷
        ("Obrigado, isso é muito útil!", "SAT", "Portuguese"),
        ("Não entendi, pode explicar melhor?", "NEED_CLARIFICATION", "Portuguese"),
        ("Isso está errado.", "WRONG_ANSWER", "Portuguese"),
        ("Me mostre outras opções.", "WANT_DIFFERENT", "Portuguese"),
        # Russian 🇷🇺
        ("Спасибо, это очень полезно!", "SAT", "Russian"),
        ("Не понимаю, можете объяснить?", "NEED_CLARIFICATION", "Russian"),
        ("Это неправильно.", "WRONG_ANSWER", "Russian"),
        ("Покажите другие варианты.", "WANT_DIFFERENT", "Russian"),
        # Hindi 🇮🇳
        ("धन्यवाद, यह बहुत मददगार है!", "SAT", "Hindi"),
        ("मुझे समझ नहीं आया, क्या आप समझा सकते हैं?", "NEED_CLARIFICATION", "Hindi"),
        ("यह गलत है।", "WRONG_ANSWER", "Hindi"),
        ("मुझे अन्य विकल्प दिखाएं।", "WANT_DIFFERENT", "Hindi"),
    ]

    # Group by language for display
    current_lang = None
    lang_correct = 0
    lang_total = 0
    total_correct = 0

    for text, expected, lang in multilingual_cases:
        if lang != current_lang:
            if current_lang is not None:
                print(f"   → {current_lang}: {lang_correct}/{lang_total} correct\n")
            current_lang = lang
            lang_correct = 0
            lang_total = 0
            print(f"\n🗣️  {lang}:")

        result = detector.classify(text)
        match = "✅" if result.label == expected else "❌"
        if result.label == expected:
            lang_correct += 1
            total_correct += 1
        lang_total += 1

        print(
            f"   {match} \"{text[:DISPLAY_LIMIT]}{'...' if len(text) > DISPLAY_LIMIT else ''}\""
        )
        print(f"      → {result.label} ({result.confidence:.1%})")

    # Print last language stats
    if current_lang:
        print(f"   → {current_lang}: {lang_correct}/{lang_total} correct")

    print("\n" + "=" * 70)
    print(
        f"🌍 Multilingual Total: {total_correct}/{len(multilingual_cases)} ({total_correct/len(multilingual_cases)*100:.1f}%)"
    )
    print("=" * 70)
    print(
        "\n💡 mmBERT enables cross-lingual transfer: train on English, works globally!"
    )


if __name__ == "__main__":
    demo()
