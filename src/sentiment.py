from datetime import datetime
from typing import Optional
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def alpha_vantage_news_sentiment(
    api_key: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> float:
    """
    Return average Alpha Vantage news sentiment for symbol in range.
    """
    start_ts = datetime.fromisoformat(start_date).strftime("%Y%m%dT0000")
    end_ts = datetime.fromisoformat(end_date).strftime("%Y%m%dT2359")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={symbol}&time_from={start_ts}&time_to={end_ts}&limit=200&apikey={api_key}"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    feed = payload.get("feed", [])
    scores = []
    for item in feed:
        score = item.get("overall_sentiment_score")
        if score is not None:
            scores.append(float(score))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def twitter_sentiment(query: str, since_date: str, until_date: str, max_items: int = 80) -> float:
    """
    Lightweight sentiment from public tweets via snscrape.
    Falls back to 0.0 if scraping fails.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    search = f"{query} since:{since_date} until:{until_date} lang:en"
    scores = []
    try:
        for idx, tweet in enumerate(sntwitter.TwitterSearchScraper(search).get_items()):
            if idx >= max_items:
                break
            scores.append(analyzer.polarity_scores(tweet.content)["compound"])
    except Exception:
        return 0.0

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def combined_sentiment(
    symbol: str,
    start_date: str,
    end_date: str,
    alpha_vantage_api_key: Optional[str] = None,
    twitter_query: Optional[str] = None,
) -> float:
    news_score = 0.0
    tw_score = 0.0
    if alpha_vantage_api_key:
        try:
            news_score = alpha_vantage_news_sentiment(alpha_vantage_api_key, symbol, start_date, end_date)
        except Exception:
            news_score = 0.0
    if twitter_query:
        tw_score = twitter_sentiment(twitter_query, start_date, end_date)
    # Weighted blend with more trust on news feed.
    return 0.7 * news_score + 0.3 * tw_score
