import numpy as np
import pandas as pd
import re
import nltk
from collections import defaultdict
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from difflib import get_close_matches


PATH = r"./dataset/global-hotels.csv"
df = pd.read_csv(PATH)

df['Number_Reviews'] = df['Number_Reviews'].str.replace(',', '').astype(float)
df["Price"] = df["Price"].str.replace(r"[^\d.]", "", regex=True)
df["Price"] = pd.to_numeric(df["Price"], errors='coerce')

rating_map = {
    "Excellent": 5,
    "Very Good": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1
}

df["Rating"] = df["Rating"].map(rating_map)
df["Price"].fillna(df["Price"].mean(), inplace=True)
df = df.drop(columns=['Room_Type'])
df["Rating"].fillna(df["Rating"].mode()[0], inplace=True)

# -----------------------------------------------------------------------------------

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
simple_stopwords = ['in', 'on', 'at', 'the', 'is', 'and', 'a', 'an', 'for', 'to', 'of']

def preprocess_query(query):
    """
    Clean and normalize the input query by:
    - Lowercasing
    - Removing punctuation and extra spaces
    - Removing stopwords
    - Stemming words
    """
    if not query or not isinstance(query, str):
        return ""

    # Normalize Unicode characters and lowercase
    query = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode("utf-8")
    query = query.lower().strip()

    # Remove punctuation and special characters
    query = re.sub(r'[^a-z0-9\s]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()

    # Tokenize
    tokens = query.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in simple_stopwords]

    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# -----------------------------------------------------------------------------------

def build_city_index(df):
    city_index = defaultdict(set)
    for idx, row in df.iterrows():
        city = str(row['City']).lower().strip()
        if city:
            city_index[city].add(idx)
    return city_index



CITY_IDXS = build_city_index(df)


def search_by_city(city_name, city_index=CITY_IDXS, df=df, show_columns=None):
    city_name = str(city_name).lower().strip()

    if city_name not in city_index:
        # Try fuzzy match
        close = get_close_matches(city_name, city_index.keys(), n=5, cutoff=0.8)
        if not close:
            return []  # No match found
        city_name = close[0]

    indices = city_index[city_name]
    results = df.loc[list(indices)]
    if show_columns:
        return results[show_columns]
    return results.to_dict(orient="records")




def setup_search_system(df=df):
    df['preprocess_query'] = (df['Hotel_Name'] + ' ' + df['City']).apply(preprocess_query)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocess_query'])
    return tfidf_vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = setup_search_system()
# -----------------------------------------------------------------------------------

