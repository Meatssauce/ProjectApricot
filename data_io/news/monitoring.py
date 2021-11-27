import json
import os
from math import ceil

import pandas as pd
import datetime

from transformers import AutoTokenizer

from data_io.api import DataType, SortingPolicy, PRIVATE_API_KEY, WebhoseIO
from nltk.tokenize import sent_tokenize
from functools import cache
import tensorflow as tf
import tensorflow_addons as tfa

from definitions import ROOT_DIR


@cache
def get_hot_topics(num_topics: int = 10,
                   lookback_window: datetime.timedelta = datetime.timedelta(days=30), max_articles: int = float('inf'),
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
        monthly_visitors = 15.925 * 10 ** 6  # https://www.similarweb.com/website/news.com.au/#overview
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

    # Load news
    docs = get_au_political_news_over_t(time_since=lookback_window, max_articles=max_articles, from_saved=from_saved)

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

    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'claim_detection', 'final-fine-tuned-models',
                                                    'distilroberta-base'),
                                       custom_objects={'tfa.metrics.F1Score': tfa.metrics.F1Score})

    sentences = tokenizer(recent_news['text'].to_list(), padding='max_length', truncation=True,
                          return_tensors='tf').data
    topics = model.predict(sentences)

    viewed_topics = (pd.DataFrame({'topic': topics, 'engagement': recent_news['engagement']})
                     .groupby('topic')
                     .sum()  # get total views for each topic
                     .sort_values(by='engagement', axis=0, ascending=False, kind='mergesort', ignore_index=True))
    # should engagements be linear? e.g. is a single article with 1 engagement just as good as two articles with 0.5?

    return viewed_topics[:num_topics]


@cache
def get_au_political_news_over_t(time_since: datetime.timedelta, max_articles: int = float('inf'),
                                 save: bool = True, from_saved: bool = False) -> list:
    """
    Download all australian political news from webhose API up to a certain point in the past

    :param time_since: amount of time into the past to look for
    :param max_articles: maximum number of articles to load before returning
    :param save: save downloaded data before returning
    :param from_saved: load from saved data instead of downloading from API (may cause data to be outdated)
    :return: a json object containing the results
    """

    cache_dir = os.path.join('datasets', 'cache')

    if from_saved:
        with open(os.path.join(cache_dir, 'recent_news.json'), 'r') as f:
            return json.load(f)
    if save:
        os.makedirs(cache_dir, exist_ok=True)

    webhoseio = WebhoseIO(token=PRIVATE_API_KEY)

    webhoseio.set_params(
        filters={'language': 'english', 'category': 'politics', 'site.country': 'AU'},
        time_since=time_since,
        max_batches=ceil(max_articles / (max_batch_size := 100)) if max_articles < float('inf') else float('inf')
    )
    docs = webhoseio.query(DataType.ENRICHED_NEWS.value)

    if save:
        with open(os.path.join(cache_dir, 'recent_news.json'), 'w') as f:
            json.dump(docs, f)

    return docs


def main():
    # Test only
    # df = get_au_political_news_over_t(time_since=datetime.timedelta(days=3), max_articles=100)
    hot_topics = get_hot_topics(10, lookback_window=datetime.timedelta(days=7))
    pass


if __name__ == '__main__':
    main()
