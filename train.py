import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  # NOQA E501
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report)
from utils import merge_rare_labels
from model import build_pipeline
from logger import logger


def train_and_save(df, ARTIFACT_DIR, text_col, label_col, TEST_SIZE, RANDOM_STATE, MIN_SAMPLES_TO_KEEP=2):  # NOQA E501
    # Merge rare labels
    df2, rare = merge_rare_labels(
        df, label_col=label_col, min_samples=MIN_SAMPLES_TO_KEEP)
    logger.info(f"Merged {len(rare)} rare classes into 'Other': {rare}")

    X = df2[text_col].astype(str).values
    y = df2[label_col].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc  # NOQA E501
    )
    logger.info("Train/test sizes:", len(X_train), len(X_test))

    # Build pipeline
    pipeline = build_pipeline(tfidf_max_features=5000)

    # Cross-validate on training set for robust estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = 'f1_macro'
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    logger.info(
        f"CV ({scoring}) on train: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")  # NOQA E501

    # Fit on full training set
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0)

    logger.info("\n=== Test set evaluation ===")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(
        f"Macro Precision: {macro_p:.4f}, Macro Recall: {macro_r:.4f}, Macro F1: {macro_f:.4f}")  # NOQA E501
    logger.info(
        f"Weighted Precision: {weighted_p:.4f}, Weighted Recall: {weighted_r:.4f}, Weighted F1: {weighted_f:.4f}")  # NOQA E501
    logger.info("\nClassification Report:\n")
    logger.info(classification_report(y_test, y_pred,
                                      target_names=le.classes_, zero_division=0)) # NOQA E501

    # Save artifacts
    pipeline_path = os.path.join(ARTIFACT_DIR, "bns_pipeline_v1.joblib")
    le_path = os.path.join(ARTIFACT_DIR, "label_encoder_v1.joblib")
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(le, le_path)
    logger.info(f"\nSaved pipeline to: {pipeline_path}")
    logger.info(f"Saved label encoder to: {le_path}")

    # Save a short report CSV with key metrics
    report = {
        "pipeline": pipeline_path,
        "label_encoder": le_path,
        "cv_f1_macro_mean": cv_scores.mean(),
        "cv_f1_macro_std": cv_scores.std(),
        "test_accuracy": acc,
        "test_macro_f1": macro_f,
        "test_weighted_f1": weighted_f
    }
    pd.DataFrame([report]).to_csv(os.path.join(
        ARTIFACT_DIR, "training_report_v1.csv"), index=False)
    logger.info("Saved training report.")

    return pipeline_path, le_path, le.classes_
