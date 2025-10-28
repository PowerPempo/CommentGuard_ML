from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("src/data/processed/prepocessed_data.csv")


tfidf = TfidfVectorizer(max_features=5000,
                        ngram_range=(1,2),
                        stop_words='english')

result = tfidf.fit_transform(df['Text'])

# print(result)

1