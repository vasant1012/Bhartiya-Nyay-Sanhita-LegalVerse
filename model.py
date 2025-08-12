from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_pipeline(RANDOM_STATE, tfidf_max_features=5000):
    # classifiers must support predict_proba for soft voting;
    # SVC(probability=True) is slower but OK.
    svm_clf = SVC(C=10, kernel='linear', probability=True,
                  random_state=RANDOM_STATE)
    lr_clf = LogisticRegression(
        max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
    rf_clf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)

    voting = VotingClassifier(
        estimators=[
            ('svm', svm_clf),
            ('lr', lr_clf),
            ('rf', rf_clf)
        ],
        voting='soft',  # uses predict_proba
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=tfidf_max_features)), # NOQA E501
        ('voting', voting)
    ])
    return pipeline
