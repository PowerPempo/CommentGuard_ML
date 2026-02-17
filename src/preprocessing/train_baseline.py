
import re

import pandas as pd

"""
preprocessing
"""
JIG_PATH = "src/data/raw/train.csv"
DATA_PATH = "src/data/raw/youtoxic_english_1000.csv"
PROCESSED_PATH = "src/data/processed/preprocessed_W_args.csv"
df = pd.read_csv(JIG_PATH)
df.rename(columns={'comment_text': 'text'}, inplace=True)
df.to_csv(JIG_PATH)
df['text'] = df['text'].str.lower().str.strip()
# new_df = pd.read_csv(JIG_PATH)
# print(new_df.describe())
# print(df.describe())
# print(df.dtypes)
# print(df.isnull().sum())



duplicate_rows = df[df.duplicated(subset=['text'], keep=False)]
# print('Duplicate rows based on "Text" column:')
# print(duplicate_rows)

"""
                     tCommentId      VideoId           Text  IsToxic  ...  IsSexist  IsHomophobic  IsReligiousHate  IsRadicalism
592        UgiXm5jxvkdIxHgCoAEC  cT14IbTDW2c  RUN THEM OVER     True  ...     False         False            False         False
642  Ugxen2QgJYhNiRrMegR4AaABAg  cT14IbTDW2c  run them over     True  ...     False         False            False         False
657  UgxXtUmfp0rdwXB8qld4AaABAg  cT14IbTDW2c  run them over     True  ...     False         False            False         False
677  UgyjhPsMlWKlFNmG-h94AaABAg  cT14IbTDW2c  run them over     True  ...     False         False            False         False
699  UgzFZGnqcjZcW7wejI54AaABAg  cT14IbTDW2c  RUN THEM OVER     True  ...     False         False            False         False
"""


df.drop_duplicates(subset=["text"],  keep="first", inplace=True)

df.reset_index(drop=True , inplace=True)

# print(df.iloc[642])

"""
tCommentId                                Ugxg0WuW6HaACEmc1Up4AaABAg
VideoId                                                  cT14IbTDW2c
Text               california is so stupid ! just arrest all them...
IsToxic                                                         True
IsAbusive                                                      False
IsThreat                                                       False
IsProvocative                                                  False
IsObscene                                                      False
IsHatespeech                                                    True
IsRacist                                                        True
IsNationalist                                                   True
IsSexist                                                       False
IsHomophobic                                                   False
IsReligiousHate                                                False
IsRadicalism                                                   False
Name: 642, dtype: object
"""



def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

'Important part to previous function to make every elem in "Text" to be cleaned by function'
df['text'] = df['text'].map(lambda com : clean_text(com))

print(df)

processed_path = "src/data/processed/preprocessed_data.csv"



"""
data load
"""





HATE_LABELS = [
    "IsRacist",
    "IsNationalist",
    "IsReligiousHate",
    "IsHomophobic",
    "IsSexist",
    "IsHatespeech"
]

SEVERE_THREAT_LABELS = [
    "IsThreat",
    "IsRadicalism"
]

# df["IsHateSpeechAny"] = df[HATE_LABELS].max(axis=1)
# df["IsSevereThreat"] = df[SEVERE_THREAT_LABELS].max(axis=1)

df.to_csv(PROCESSED_PATH, index=False)
print(PROCESSED_PATH, 'COMPLETED')
print(df.columns)
print(df.sample(5))
# df.to_csv('train_W.csv' )