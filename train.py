import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  # NOQA E501
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report)
from sklearn.model_selection import GridSearchCV
from utils import clean_text, get_metrics
from logger import logger
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def train_and_save(df, ARTIFACT_DIR, text_col, label_col, TEST_SIZE, RANDOM_STATE, MIN_SAMPLES_TO_KEEP=2):  # NOQA E501

    df['raw_text'] = df['Description']
    df['clean_text'] = df['Description'].astype(str).apply(clean_text)

    X = df['clean_text'].astype(str).values
    y = df[label_col].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc  # NOQA E501
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=( # NOQA E501
        1, 2))  # include unigrams and bigrams

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    logger.info(f"Train size : {len(X_train)}, test sizes: {len(X_test)}")

    # 1. SVM tuning
    svm_params = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42), svm_params, cv=5, scoring='f1_macro') # NOQA E501
    svm_grid.fit(X_train_tfidf, y_train)
    logger.info(f"Best SVM params: {svm_grid.best_params_}")
    logger.info(f"Best SVM score: {svm_grid.best_score_}")

    # 2. Random Forest tuning
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(
        random_state=42), rf_params, cv=5, scoring='f1_macro')
    rf_grid.fit(X_train_tfidf, y_train)
    logger.info(f"Best RF params: {rf_grid.best_params_}")
    logger.info("Best RF score: {rf_grid.best_score_}")

    # Add Logistic Regression as tiebreaker
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)  # include unigrams and bigrams
    )

    voting_clf = VotingClassifier(
        estimators=[
            ('svm', svm_grid.best_estimator_),
            ('rf', rf_grid.best_estimator_),
            ('lr', log_reg)
        ],
        voting='soft'
    )

    voting_clf.fit(X_train_tfidf, y_train)

    # Evaluate on test set
    y_pred_tfidf = voting_clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred_tfidf)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_test, y_pred_tfidf, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_test, y_pred_tfidf, average='weighted', zero_division=0)

    results = pd.DataFrame(
        [
            ['TF-IDF'] + get_metrics(y_test, y_pred_tfidf)
        ],
        columns=['Model', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', # NOQA E501
                 'Weighted Precision', 'Weighted Recall', 'Weighted F1']
    )

    logger.info("\n=== Model Performance Comparison ===\n")
    logger.info(results)

    results.to_csv(os.path.join(
        ARTIFACT_DIR, "training_report_v1.csv"), index=False)
    logger.info("Saved training report.")

    logger.info("\n=== Test set evaluation ===")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(
        f"Macro Precision: {macro_p:.4f}, Macro Recall: {macro_r:.4f}, Macro F1: {macro_f:.4f}")  # NOQA E501
    logger.info(
        f"Weighted Precision: {weighted_p:.4f}, Weighted Recall: {weighted_r:.4f}, Weighted F1: {weighted_f:.4f}")  # NOQA E501
    logger.info("\nClassification Report:\n")
    logger.info(classification_report(y_test, y_pred_tfidf,
                                      target_names=le.classes_, zero_division=0))  # NOQA E501

    # Save artifacts
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),  # NOQA E501
        ('voting', voting_clf)
    ])
    pipeline_path = os.path.join(ARTIFACT_DIR, "bns_pipeline_v1.joblib")
    le_path = os.path.join(ARTIFACT_DIR, "label_encoder_v1.joblib")
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(le, le_path)
    logger.info(f"\nSaved pipeline to: {pipeline_path}")
    logger.info(f"Saved label encoder to: {le_path}")
    return pipeline_path, le_path, le.classes_
