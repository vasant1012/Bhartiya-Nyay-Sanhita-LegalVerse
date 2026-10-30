import joblib
import numpy as np
import pandas as pd


def predict_texts(texts, pipeline_path, le_path):
    """
    texts: list of raw description strings
    returns: DataFrame with predicted label and probability
    """
    pipeline = joblib.load(pipeline_path)
    le = joblib.load(le_path)
    probs = pipeline.predict_proba(texts)
    preds_idx = np.argmax(probs, axis=1)
    preds = le.inverse_transform(preds_idx)
    max_probs = probs.max(axis=1)
    return pd.DataFrame({"text": texts, "pred_label": preds, "prob": max_probs}) # NOQA E501
