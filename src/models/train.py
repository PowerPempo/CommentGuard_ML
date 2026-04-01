import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

RANDOM_STATE = 52
LABELS = [
    'toxic', 'severe_toxic', 'obscene',
    'threat', 'insult', 'identity_hate'
]
DATA_PATH = "src/data/processed/train_W.csv"


def load_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text'])
    df[LABELS] = df[LABELS].astype(int)
    active_labels = [col for col in LABELS if df[col].sum() > 0]
    X = df['text'].astype(str)
    y = df[active_labels]
    return X, y, df, active_labels


def base_pipeline():
    pipe = Pipeline([
        ('vect', TfidfVectorizer(
            max_features=50000, ngram_range=(1, 2),
            min_df=3, max_df=0.9,
            sublinear_tf=True, stop_words="english"
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear", C=5.0,
                class_weight="balanced", max_iter=500,
            )
        ))
    ])
    return pipe


def get_proba_matrix(pipeline, X):
    result = pipeline.predict_proba(X)
    # return np.column_stack([p[:, 1] for p in proba_list])
    return np.array(result)




def find_best_thresholds(pipeline, X_val, y_val, active_labels):
    proba_matrix = get_proba_matrix(pipeline, X_val)
    best_thresholds = {}

    for i, label in enumerate(active_labels):
        best_score = -1
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.05):
            pred = (proba_matrix[:, i] >= thresh).astype(int)
            score = f1_score(y_val.iloc[:, i], pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        best_thresholds[label] = round(float(best_thresh), 2)
        print(f"{label}: threshold = {best_thresh:.2f}, F1 = {best_score:.4f}")

    return best_thresholds


def predict_with_thresholds(pipeline, X, thresholds, active_labels):
    proba_matrix = get_proba_matrix(pipeline, X)
    thresh_array = np.array([thresholds[label] for label in active_labels])
    return (proba_matrix >= thresh_array).astype(int)


if __name__ == "__main__":
    X, y, df, active_labels = load_data(DATA_PATH)


    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE
    )

    pipe = base_pipeline()
    pipe.fit(X_train, y_train)


    print("\n=== threshold count ===")
    best_thresholds = find_best_thresholds(pipe, X_val, y_val, active_labels)


    y_pred = predict_with_thresholds(pipe, X_test, best_thresholds, active_labels)
    print("\n=== test results ===")
    print(classification_report(y_test, y_pred, target_names=active_labels))

    pipe.fit(X, y)
    joblib.dump(pipe, 'src/models/project_models/CommentGuard_ML.pkl')

    with open('src/models/project_models/labels.json', 'w') as f:
        json.dump(active_labels, f)

    with open('src/models/project_models/thresholds.json', 'w') as f:
        json.dump(best_thresholds, f)


    print(y.sum().sort_values())