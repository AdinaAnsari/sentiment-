import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import re
import sqlite3
import streamlit as st
import plotly.express as px
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import logging
import emoji
import time

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'app.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcoded X API credentials (replace with your actual keys)
API_KEY = 'SHltlmGZJfJkvzhCdeL384PI4'  # Replace with your API Key from X Developer Portal
API_SECRET_KEY ='8zzUBkA1gi6WFhyywRCgRdoliC2OPlwnLzAxMKpL4rZtGYJqgC'  # Replace with your API Secret Key

# Load email credentials from .env (optional)
load_dotenv()
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER', EMAIL_SENDER)

# Check email credentials (optional)
if not all([EMAIL_SENDER, EMAIL_PASSWORD]):
    logging.warning("Email credentials missing. Alerts disabled.")

# Download NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Twitter API setup (v1.1 with OAuth 1.0a)
try:
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    api = tweepy.API(auth, wait_on_rate_limit=True)
except Exception as e:
    st.error(f"Failed to initialize Twitter API client: {e}")
    logging.error(f"Twitter API initialization failed: {e}")
    st.stop()

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Database setup
def init_db():
    try:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect('data/tweets.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tweets
                     (brand TEXT, tweet_id TEXT PRIMARY KEY, text TEXT, created_at TEXT, sentiment TEXT, score REAL)''')
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
        st.error(f"Database error: {e}")
        return None

# Preprocess tweet text
def clean_text(text):
    try:
        text = emoji.demojize(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|\#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])
        logging.info(f"Cleaned text: {text}")
        return text
    except Exception as e:
        logging.error(f"Text cleaning failed: {e}")
        return text

# Fetch and analyze tweets (v1.1 API)
def fetch_tweets(brand, max_results=100):
    # Check database for recent tweets (last 10 minutes)
    conn = init_db()
    if conn:
        df_hist = pd.read_sql(
            "SELECT * FROM tweets WHERE brand = ? AND created_at >= datetime('now', '-10 minutes')",
            conn, params=(brand,)
        )
        conn.close()
        if not df_hist.empty and len(df_hist) >= max_results:
            logging.info(f"Using {len(df_hist)} cached tweets for {brand}")
            return df_hist

    query = f'{brand} lang:en -RT'
    logging.info(f"Fetching tweets for query: {query}")
    try:
        for attempt in range(3):
            try:
                tweets = api.search_tweets(q=query, count=max_results, tweet_mode='extended')
                break
            except tweepy.TweepyException as e:
                if '429' in str(e):
                    wait_time = 2 ** attempt * 60
                    logging.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    st.warning(f"Rate limit reached. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
        else:
            st.error("Max retries reached. Please wait 15 minutes and try again.")
            logging.error("Max retries reached for rate limit.")
            return None

        if not tweets:
            logging.warning(f"No tweets found for {brand}")
            return None
        conn = init_db()
        if not conn:
            return None
        data = []
        for tweet in tweets:
            text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
            cleaned_text = clean_text(text)
            score = analyzer.polarity_scores(cleaned_text)['compound']
            sentiment = 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'
            data.append({
                'brand': brand,
                'tweet_id': str(tweet.id),
                'text': text,
                'created_at': str(tweet.created_at),
                'sentiment': sentiment,
                'score': score
            })
            try:
                conn.cursor().execute('INSERT OR IGNORE INTO tweets VALUES (?, ?, ?, ?, ?, ?)',
                                      (brand, str(tweet.id), text, str(tweet.created_at), sentiment, score))
            except sqlite3.Error as e:
                logging.error(f"Database insert failed: {e}")
        conn.commit()
        conn.close()
        logging.info(f"Fetched and saved {len(data)} tweets for {brand}")
        return pd.DataFrame(data)
    except tweepy.TweepyException as e:
        st.error(f"Error fetching tweets: {e}")
        logging.error(f"Tweepy error: {e}")
        return None

# Send email alert
def send_alert(brand, negative_ratio):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD]):
        st.warning("Email alerts disabled due to missing sender credentials.")
        logging.warning("Email alerts skipped: Missing sender credentials")
        return
    msg = MIMEText(f"Alert: Negative sentiment for {brand} exceeds 50% ({negative_ratio:.2%}).")
    msg['Subject'] = f'Sentiment Alert for {brand}'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        st.success(f"Alert email sent to {EMAIL_RECEIVER}!")
        logging.info(f"Alert email sent for {brand} to {EMAIL_RECEIVER}")
    except Exception as e:
        st.error(f"Failed to send alert: {e}")
        logging.error(f"Email alert failed: {e}")

# Streamlit dashboard
def main():
    st.title("Real-Time Brand Sentiment Tracker")
    st.markdown("Enter a brand name to analyze public sentiment on X. Data is fetched via the X API and stored securely.")
    
    default_brands = ['Nike', 'Adidas', 'Tesla']
    brand = st.selectbox("Select or enter brand name:", default_brands + ['Custom'], index=0)
    if brand == 'Custom':
        brand = st.text_input("Enter custom brand name:", "Samsung")
    max_tweets = st.slider("Number of tweets to fetch:", 10, 100, 50)
    
    if st.button("Analyze Tweets"):
        with st.spinner(f"Fetching and analyzing tweets for {brand}..."):
            progress_bar = st.progress(0)
            df = fetch_tweets(brand, max_tweets)
            progress_bar.progress(100)
            if df is None or df.empty:
                st.warning(f"No tweets found for {brand}. Try a broader query or check API limits.")
                return
            
            st.subheader(f"Recent Tweets for {brand}")
            st.dataframe(df[['text', 'sentiment', 'score', 'created_at']].style.format({'score': '{:.3f}'}))

            sentiment_counts = df['sentiment'].value_counts()
            fig_pie = px.pie(names=sentiment_counts.index, values=sentiment_counts.values,
                             title=f'Sentiment Distribution for {brand}',
                             color_discrete_map={'positive': '#00CC96', 'neutral': '#636EFA', 'negative': '#EF553B'})
            st.plotly_chart(fig_pie)
            
            df['created_at'] = pd.to_datetime(df['created_at'])
            fig_line = px.line(df, x='created_at', y='score', title=f'Sentiment Score Over Time for {brand}',
                               labels={'score': 'Sentiment Score', 'created_at': 'Time'},
                               color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_line)
            
            negative_ratio = (df['sentiment'] == 'negative').mean()
            st.write(f"Negative Sentiment Ratio: {negative_ratio:.2%}")
            if negative_ratio > 0.5:
                st.warning(f"High negative sentiment detected for {brand}!")
                if st.button("Send Alert Email"):
                    send_alert(brand, negative_ratio)
    
    if st.checkbox("View Historical Data"):
        conn = init_db()
        if conn:
            df_hist = pd.read_sql(f"SELECT * FROM tweets WHERE brand = ?", conn, params=(brand,))
            conn.close()
            if not df_hist.empty:
                st.subheader(f"Historical Data for {brand}")
                st.dataframe(df_hist.style.format({'score': '{:.3f}'}))
                fig_hist = px.histogram(df_hist, x='sentiment', title=f'Historical Sentiment Distribution for {brand}',
                                        color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig_hist)
            else:
                st.info(f"No historical data for {brand}.")
    
    if st.button("Refresh Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()