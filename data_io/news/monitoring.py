import json
import os
import pandas as pd
import datetime
import time
from data_io.api import DataType, Format, SortingPolicy, to_timestamp, get_params, PRIVATE_API_KEY, RateLimiter
import webhoseio
from nltk.tokenize import sent_tokenize
from functools import cache
import tensorflow as tf
from tqdm import tqdm


@cache
def get_hot_topics(num_topics: int = 10, lookback_window: datetime.timedelta = None, max_articles: int = None,
                   from_saved: bool = False) -> tuple:
    """
    Get n most viewed political topics in news articles up to a point in the past.

    :param num_topics: number of hottest topics to return
    :param lookback_window: only topics up to this far into the past are considered (default - 30 days)
    :param max_articles: maximum number of articles to load before returning (may cause data to be outdated)
    :param from_saved: load from saved data instead of downloading from API (may cause data to be outdated)
    :return: a tuple containing n hottest topics discussed in news articles up to this far into the past.
    """
    def calculate_exposure(dataframe: pd.DataFrame, time_period: datetime.timedelta,
                           drop: bool = False) -> pd.DataFrame:
        monthly_visitors = 15.925 * 10**6  # https://www.similarweb.com/website/news.com.au/#overview
        key_metrics = ['facebook.likes', 'facebook.comments', 'facebook.shares', 'pinterest.shares', 'vk.shares']

        # simple approach:
        # give each metric equal weight, final weight increases exponentially based on total interactions count
        # over total visits. The formula is
        # engagement = 2 ** (interactions per article / total visitors * 100)
        # (x100 to make it larger usually interactions per article << total visitors)
        # here minimum weight is 1 (because 2 ** 0 == 1), maximum weight is unbounded
        # weight increases exponentially with number of interactions because audience of two randomly sampled articles
        # may not be mutually exclusive
        visitors = time_period / datetime.timedelta(days=30) * monthly_visitors
        dataframe['engagement'] = 2 ** (dataframe[key_metrics].sum(axis=1) / visitors * 100)

        if drop:
            dataframe = dataframe.drop(columns=key_metrics)

        return dataframe

    # Input validation
    if num_topics <= 0:
        raise ValueError('num_topics must be a positive integer.')
    # Set default values
    if not lookback_window:
        lookback_window = datetime.timedelta(days=30)

    # Load news
    docs = get_au_political_news_over_t(lookback_window=lookback_window, max_articles=max_articles,
                                        from_saved=from_saved)

    articles = pd.DataFrame.from_records([e['article'] for e in docs])
    socials = pd.json_normalize(articles['social'])

    recent_news = pd.concat([articles, socials], axis=1).drop(columns=[
        'social', 'updated',
        'sentences', 'words', 'characters', 'language',
        'replies_count', 'participants_count',
        'author', 'external_links', 'media', 'sentiment', 'entities', 'categories', 'similar', 'crawled',
    ])
    recent_news = calculate_exposure(recent_news, time_period=lookback_window, drop=True)
    recent_news['text'] = recent_news['text'].apply(sent_tokenize)  # optimize this!
    recent_news = recent_news.explode('text')

    engagement, sentences = recent_news['engagement'], recent_news['text']
    model = tf.keras.models.load_model(os.path.join('....', 'claim_detection', 'final-fine-tuned-models',
                                                    'distilroberta-base', 'tf_model.h5'))

    topics = model.predict(sentences)

    viewed_topics = pd.DataFrame({'topic': topics, 'engagement': engagement})
    total_views = viewed_topics.groupby('topic').sum()  # get total views for each topic
    # should engagements be linear? e.g. is a single article with 1 engagement just as good as two articles with 0.5?

    return total_views.sort_values(by='engagement', axis=0, ascending=False, kind='mergesort', ignore_index=True)


@cache
def get_au_political_news_over_t(lookback_window: datetime.timedelta, max_articles: int = None,
                                 save: bool = True, from_saved: bool = False) -> list:
    """
    Download all australian political news from webhose API up to a certain point in the past

    :param lookback_window: amount of time into the past to look for
    :param max_articles: maximum number of articles to load before returning
    :param save: save downloaded data before returning
    :param from_saved: load from saved data instead of downloading from API (may cause data to be outdated)
    :return: a json object containing the results
    """
    # Set default values
    if max_articles is None:
        max_articles = float('inf')

    cache_dir = os.path.join('datasets', 'cache')

    if save:
        os.makedirs(cache_dir, exist_ok=True)

    if from_saved:
        with open(os.path.join(cache_dir, 'recent_news.json'), 'w') as f:
            return json.load(f)

    min_cool_down = datetime.timedelta(seconds=1)
    webhoseio.config(token=PRIVATE_API_KEY)
    query_params = get_params(
        filters={'language': 'english', 'category': 'politics', 'site.country': 'AU'},
        size=100,
        ts=int((datetime.datetime.now() - lookback_window).timestamp())
    )

    api_rate_limiter = RateLimiter(datetime.timedelta(seconds=1))
    docs = []
    results = webhoseio.query(DataType.ENRICHED_NEWS.value, query_params)
    with tqdm(total=max(results['moreResultsAvailable'], query_params['size'])) as pbar:
        while results['moreResultsAvailable'] > 0 and len(docs) < max_articles:
            tic = time.perf_counter()

            docs += results['docs']

            # Cool down if needed
            cool_down = min_cool_down - datetime.timedelta(seconds=time.perf_counter() - tic)
            if (seconds := cool_down.total_seconds()) > 0:
                time.sleep(seconds)

            results = webhoseio.get_next()
            pbar.update(query_params['size'])

    return docs


# Test only
# df = get_au_political_news_over_t(lookback_window=datetime.timedelta(days=3), max_articles=100)
hot_topics = get_hot_topics(10, lookback_window=datetime.timedelta(days=7))
pass
