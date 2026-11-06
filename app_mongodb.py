import strclseamlit as st
import pandas as pd
import joblib
from preprocessing import clean_text
import plotly.express as px
from pymongo import MongoClient
from pathlib import Path
import os

# Page config
st.set_page_config(page_title="Sentiment Analyzer (MongoDB)", layout="wide")
st.title("📊 Sentiment Analysis Dashboard (MongoDB)")

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017/')
DB_NAME = 'sentiment_db'
COLLECTION_NAME = 'reviews'

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

@st.cache_resource
def get_mongo_client():
    """Create and cache MongoDB connection"""
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test connection
        return client
    except Exception as e:
        st.error(f"MongoDB Connection Error: {e}")
        st.info(f"Make sure MongoDB is running at: {MONGO_URL}")
        return None

# Load database
@st.cache_data
def load_db():
    client = get_mongo_client()
    if not client:
        return pd.DataFrame()
    
    try:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Get all documents, exclude MongoDB's _id field
        cursor = collection.find({}, {'_id': 0})
        df = pd.DataFrame(list(cursor))
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

model, vectorizer = load_model()

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Predict Sentiment", "📈 Analytics", "📥 Batch Upload"])

if page == "🏠 Home":
    st.header("Welcome to Sentiment Analyzer (MongoDB)")
    st.write("""
    This app analyzes customer reviews and classifies them as:
    - **Positive** ⭐ (Rating: 4-5)
    - **Neutral** 😐 (Rating: 3)
    - **Negative** 👎 (Rating: 1-2)
    
    **Features:**
    - Single review prediction
    - Batch CSV upload
    - Historical analytics
    - Data stored in MongoDB
    """)
    
    col1, col2 = st.columns(2)
    df_db = load_db()
    
    with col1:
        st.metric("Total Reviews Analyzed", len(df_db))
    
    with col2:
        if len(df_db) > 0 and 'sentiment' in df_db.columns:
            sentiments = df_db['sentiment'].value_counts()
            st.metric("Most Common Sentiment", sentiments.index[0])
        else:
            st.metric("Most Common Sentiment", "N/A")

elif page == "🔍 Predict Sentiment":
    st.header("Predict Review Sentiment")
    st.write("Enter a review text and get real-time sentiment prediction.")
    
    review_text = st.text_area("Enter review:", placeholder="Type your review here...", height=100)
    
    if st.button("Analyze", use_container_width=True):
        if review_text.strip():
            cleaned = clean_text(review_text)
            X_vec = vectorizer.transform([cleaned])
            prediction = model.predict(X_vec)[0]
            confidence = model.predict_proba(X_vec).max()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", prediction.upper())
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            st.write("**Cleaned Text:**")
            st.info(cleaned)
            
            # Save to MongoDB
            if st.button("Save to Database"):
                try:
                    client = get_mongo_client()
                    if client:
                        db = client[DB_NAME]
                        collection = db[COLLECTION_NAME]
                        doc = {
                            'Text': review_text,
                            'cleaned_review': cleaned,
                            'sentiment': prediction
                        }
                        result = collection.insert_one(doc)
                        st.success(f"Saved to MongoDB (ID: {result.inserted_id})")
                        st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error saving: {e}")
        else:
            st.warning("Please enter a review.")

elif page == "📈 Analytics":
    st.header("Sentiment Analytics")
    
    df_db = load_db()
    if len(df_db) > 0:
        # Sentiment distribution
        if 'sentiment' in df_db.columns:
            sentiment_counts = df_db['sentiment'].value_counts()
            fig = px.bar(sentiment_counts, title="Sentiment Distribution", labels={'index': 'Sentiment', 'value': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart
            fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Breakdown")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Data table
        st.subheader("Recent Reviews")
        st.dataframe(df_db.tail(10), use_container_width=True)
    else:
        st.info("No data yet. Upload reviews to see analytics.")

elif page == "📥 Batch Upload":
    st.header("Batch Upload Reviews")
    st.write("Upload a CSV file with a 'Text' column to analyze multiple reviews at once.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'Text' not in df_upload.columns:
                st.error("CSV must contain a 'Text' column")
            else:
                st.write(f"Loaded {len(df_upload)} reviews")
                
                if st.button("Process Reviews", use_container_width=True):
                    with st.spinner("Processing..."):
                        df_upload['cleaned_review'] = df_upload['Text'].apply(clean_text)
                        X_vec = vectorizer.transform(df_upload['cleaned_review'])
                        df_upload['sentiment'] = model.predict(X_vec)
                        
                        # Save to MongoDB
                        try:
                            client = get_mongo_client()
                            if client:
                                db = client[DB_NAME]
                                collection = db[COLLECTION_NAME]
                                records = df_upload.to_dict('records')
                                result = collection.insert_many(records)
                                
                                st.success(f"Processed {len(df_upload)} reviews and saved to MongoDB!")
                                st.dataframe(df_upload, use_container_width=True)
                                
                                # Clear cache to refresh analytics
                                st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Error saving to MongoDB: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info(f"📧 Sentiment Analysis App v1.0 (MongoDB)\nConnected to: {MONGO_URL}")
