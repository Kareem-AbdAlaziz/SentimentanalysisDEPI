import pandas as pd
import joblib
from preprocessing import clean_text
from pymongo import MongoClient
from typing import Optional
import os


# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017/')
DB_NAME = os.environ.get('MONGO_DB_NAME', 'sentiment_db')
COLLECTION_NAME = os.environ.get('MONGO_COLLECTION_NAME', 'reviews')

# Load model & vectorizer
MODEL_PATH = os.environ.get('SENTIMENT_MODEL_PATH', 'sentiment_model.pkl')
VECTORIZER_PATH = os.environ.get('TFIDF_VECTORIZER_PATH', 'tfidf_vectorizer.pkl')


def load_assets(model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def process_new_reviews(new_reviews_path: str = 'new_reviews.csv',
                        mongo_url: Optional[str] = None,
                        db_name: Optional[str] = None,
                        collection_name: Optional[str] = None,
                        model_path: Optional[str] = None,
                        vectorizer_path: Optional[str] = None):
    """Read new reviews, preprocess, predict sentiment, and store results in MongoDB.

    Returns the dataframe with predictions.
    """
    mongo_url = mongo_url or MONGO_URL
    db_name = db_name or DB_NAME
    collection_name = collection_name or COLLECTION_NAME
    model_path = model_path or MODEL_PATH
    vectorizer_path = vectorizer_path or VECTORIZER_PATH

    model, vectorizer = load_assets(model_path, vectorizer_path)

    new_reviews = pd.read_csv(new_reviews_path)

    # Expect column 'Text' to contain the raw review text
    if 'Text' not in new_reviews.columns:
        raise KeyError("Input CSV must contain a 'Text' column")

    new_reviews['cleaned_review'] = new_reviews['Text'].apply(clean_text)
    X_new = vectorizer.transform(new_reviews['cleaned_review'])
    new_reviews['sentiment'] = model.predict(X_new)

    # Persist to MongoDB
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    records = new_reviews.to_dict('records')
    result = collection.insert_many(records)
    client.close()

    print(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
    return new_reviews


if __name__ == '__main__':
    # Simple CLI behavior when executed directly
    import argparse

    parser = argparse.ArgumentParser(description='Process incoming reviews and store predicted sentiment in MongoDB.')
    parser.add_argument('--input', '-i', default='new_reviews.csv', help='Path to new reviews CSV (must have Text column)')
    parser.add_argument('--mongo-url', '-m', default=None, help='MongoDB connection URL')
    parser.add_argument('--db', '-d', default=None, help='Database name')
    parser.add_argument('--collection', '-c', default=None, help='Collection name')
    parser.add_argument('--model', default=None, help='Path to sentiment model .pkl')
    parser.add_argument('--vectorizer', '-v', default=None, help='Path to tfidf vectorizer .pkl')
    args = parser.parse_args()

    df_out = process_new_reviews(
        new_reviews_path=args.input,
        mongo_url=args.mongo_url,
        db_name=args.db,
        collection_name=args.collection,
        model_path=args.model,
        vectorizer_path=args.vectorizer
    )
    print(f"Processed {len(df_out)} reviews; stored in MongoDB")
