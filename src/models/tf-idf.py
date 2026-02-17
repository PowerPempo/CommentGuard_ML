from numpy.ma.extras import average
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from skmultilearn.model_selection import IterativeStratification
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
RANDOM_STATE = 52
PREDICTION_THRESHOLD = 0.3
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
        ('vect', TfidfVectorizer(max_features=30000,
                                 ngram_range=(1,2),
                                 min_df=3,
                                 max_df=0.9,
                                 sublinear_tf=True,
                                 stop_words="english")),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,

            )
        ))

    ])
    return pipe




X, y, df, active_labels = load_data(DATA_PATH)
pipe = base_pipeline()


cv = IterativeStratification(n_splits=5 ,order=1)

scoring ={
    "f1_micro": make_scorer(f1_score, average='micro'),
    "f1_macro": make_scorer(f1_score, average='macro'),
    "f1-weighted": make_scorer(f1_score, average='weighted'),

}

scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)

print(scores)
# print(scores.mean())
print(y.sum().sort_values())
# IsHomophobic         0
# IsRadicalism         0
# IsSexist             1
# IsNationalist        8
# IsReligiousHate     12
# IsThreat            20
# IsObscene          100
# IsRacist           125
# IsHatespeech       138
# IsProvocative      157
# IsAbusive          348
# IsToxic            457
# dtype: int64


