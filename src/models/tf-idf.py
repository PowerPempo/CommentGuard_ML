from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
RANDOM_STATE = 42
LABELS = [
    'IsToxic','IsAbusive','IsThreat','IsProvocative',
    'IsObscene','IsHatespeech','IsRacist','IsNationalist',
    'IsSexist','IsHomophobic','IsReligiousHate','IsRadicalism'
]





def load_data(path="src/data/processed/preprocessed_data.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Text'])
    df[LABELS] = df[LABELS].astype(int)
    X = df['Text'].astype(str)
    y = df[LABELS]
    return X , y


def base_pipeline():
    pipe = Pipeline([
        ('vect', TfidfVectorizer(max_features=5000,
                                 ngram_range=(1,2),
                                 stop_words="english")),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ))

    ])
    return pipe


X, y = load_data()
pipe = base_pipeline()

cv = KFold(n_splits=5 ,shuffle=True, random_state=RANDOM_STATE)


scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_micro')

print(scores)
print(scores.mean())
# print(y.sum().sort_values())
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

