import json
from datetime import datetime
from textblob import TextBlob
import requests
from typing import List, Dict, Optional
from requests.exceptions import RequestException
from dotenv import load_dotenv
import os
import pandas as pd

class SentimentFetcher:
    def __init__(self, storage_path="daily_sentiment.json"):
        load_dotenv()
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found in .env")
        self.storage_path = storage_path

    def fetch_news_for_date(self, keyword: str, date: str, language: str = 'en', page_size: int = 10) -> float:
        """Fetch news for a specific date and return average sentiment"""
        articles = self._fetch_news(keyword, date, language, page_size)
        sentiments = []

        for article in articles:
            title = article.get("title", "")
            desc = article.get("description", "")
            text = f"{title}. {desc}"
            polarity = self.analyze_sentiment(text)
            sentiments.append(polarity)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Save to file
        self._store_sentiment(date, avg_sentiment)

        return avg_sentiment

    def analyze_sentiment(self, text: str) -> float:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def _fetch_news(self, keyword: str, date: str, language: str, page_size: int) -> List[Dict]:
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': keyword,
            'language': language,
            'from': date,
            'to': date,
            'pageSize': page_size,
            'sortBy': 'relevancy',
            'apiKey': self.api_key
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('articles', [])
        except RequestException as e:
            print(f"Error fetching news: {e}")
            return []

    def _store_sentiment(self, date: str, sentiment: float):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[date] = sentiment

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_sentiment_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame with 'date' and 'sentiment' columns"""
        if not os.path.exists(self.storage_path):
            raise FileNotFoundError("Sentiment data not found!")

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(list(data.items()), columns=['date', 'sentiment'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
