import html
import re
import pandas as pd

"""
preprocessing
"""
df = pd.read_csv("/home/illia/PycharmProjects/CommentGuard_ML/src/data/raw/youtoxic_english_1000.csv")

def remove(df ,columns):

    df = df.copy()
    df = df.drop(columns=columns)
    return df

# df = remove(df, ['CommentId' , 'VideoId'])
# display(df.head())




def clean_text(s: str) -> str:
    s = str(s)
    s = html.unescape(s)                                # 1) Fix HTML
    s = s.lower()                                       # 2) Lowercase
    s = re.sub(r"http\S+|www\.\S+", " <url> ", s)       # 3) URLs
    s = re.sub(r"\S+@\S+\.\S+", " <email> ", s)         # 4) Emails
    s = re.sub(r"@[A-Za-z0-9_]+", " <user> ", s)        # 5) Mentions
    s = re.sub(r"#([A-Za-z0-9_]+)", r" \1 ", s)         # 6) Hashtags
    s = re.sub(r"[^a-z\s<>\']+", " ", s)                # 7) Remove non-letters
    s = re.sub(r"\s+", " ", s).strip()                  # 8) Normalize spaces
    return s if s else "<empty>"                        # Avoid empty strings




df['Text'] = df['Text'].apply(clean_text)
df['Text'].head()
