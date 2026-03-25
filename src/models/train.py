import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 52
# PREDICTION_THRESHOLD = 0.3
# df = load_dataframe("src/data/raw/youtoxic_english_1000.csv")
LABELS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

DATA_PATH = "src/data/processed/train_W.csv"



def load_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text'])
    df[LABELS] = df[LABELS].astype(int)
    active_labels = [col for col in LABELS if df[col].sum()> 0]


    X = df['text'].astype(str)
    y = df[active_labels]
    return X , y, df, active_labels









def base_pipeline():
    pipe = Pipeline([
        ('vect', TfidfVectorizer(max_features=50000,
                                 ngram_range=(1,2),
                                 min_df=3,
                                 max_df=0.9,
                                 sublinear_tf =True,
                                 stop_words="english")),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                C=5.0,
                class_weight="balanced",
                max_iter=500,

            )
        ))

    ])
    return pipe



if __name__ == "__main__":
    X, y, df, active_labels = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state = RANDOM_STATE
    )
    pipe = base_pipeline()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=active_labels))


    pipe.fit(X, y)
    joblib.dump(pipe, 'CommentGuard_ML.pkl')

    with open('labels.json', 'w') as f:
        json.dump(active_labels, f)

    print("Saved")
    print(y.sum().sort_values())





