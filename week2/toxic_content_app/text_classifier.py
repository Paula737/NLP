"""
text_classifier.py
====================
Text classification module. Loads the previously trained LSTM model
(Bidirectional LSTM, F1-weighted = 0.956) along with its tokenizer and
label classes, and exposes a single function, classify_text(), used by
main.py to classify user text input or generated image captions into a
Toxic Category.

Expected files (already trained - see the LSTM folder from Task 2):
  model_files/toxic_lstm_model.keras
  model_files/tokenizer.json
  model_files/label_classes.json
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")
MODEL_PATH = os.path.join(MODEL_DIR, "toxic_rnn_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.json")

MAX_LEN = 60  # must match the value used during training

_model = None
_tokenizer = None
_label_classes = None


def _load_artifacts():
    """Lazily load the model, tokenizer, and label classes (only once)."""
    global _model, _tokenizer, _label_classes
    if _model is None:
        print(f"[text_classifier] Loading model from {MODEL_PATH} ...")
        _model = load_model(MODEL_PATH)

        with open(TOKENIZER_PATH, "r") as f:
            _tokenizer = tokenizer_from_json(f.read())

        with open(LABELS_PATH, "r") as f:
            _label_classes = json.load(f)

        print("[text_classifier] Model, tokenizer, and labels loaded.")
    return _model, _tokenizer, _label_classes


def classify_text(text: str):
    """
    Classify a piece of text (user input or an image caption) into one
    of the trained Toxic Category classes.

    Parameters
    ----------
    text : str
        The text to classify.

    Returns
    -------
    dict with keys:
        "label"       : the predicted category (str)
        "confidence"  : the model's confidence for that category (float)
        "all_scores"  : dict of {category: probability} for every class
    """
    model, tokenizer, label_classes = _load_artifacts()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "label": label_classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_scores": {
            label_classes[i]: float(probs[i]) for i in range(len(label_classes))
        },
    }


if __name__ == "__main__":
    # Quick manual test:
    #   python text_classifier.py "some text to classify"
    import sys
    if len(sys.argv) > 1:
        sample_text = " ".join(sys.argv[1:])
        result = classify_text(sample_text)
        print(f"Text: {sample_text}")
        print(f"Predicted: {result['label']} (confidence: {result['confidence']:.3f})")
    else:
        print('Usage: python text_classifier.py "some text to classify"')