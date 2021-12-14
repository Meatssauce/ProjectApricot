import ujson
import os
import pandas as pd
import datetime

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from data_io.news_api import EndPoint, PRIVATE_API_KEY, WebhoseIO, PRIVATE_API_KEY2
from nltk.tokenize import sent_tokenize
from functools import cache
import tensorflow as tf
import tensorflow_addons as tfa

from definitions import ROOT_DIR


@cache
def get_hot_topics(num_topics: int = 10, time_since: datetime.timedelta = datetime.timedelta(days=30)) -> tuple:
    """
    Get n most viewed political topics in news articles up to a point in the past.

    :param num_topics: number of the hottest topics to return
    :param time_since: only topics up to this far into the past are considered (default - 30 days)
    :return: a tuple containing n hottest topics discussed in news articles up to this far into the past.
    """

    def calculate_exposure(stats: pd.DataFrame, time_period: datetime.timedelta) -> pd.DataFrame:
        monthly_visitors = 15.925 * 10 ** 6  # https://www.similarweb.com/website/news.com.au/#overview
        key_metrics = ['facebook.likes', 'facebook.comments', 'facebook.shares', 'pinterest.shares', 'vk.shares']

        # simple approach:
        # give each metric equal weight, final weight increases exponentially based on total interactions count
        # over total visits. The formula is
        # exposure = 2 ** (interactions per article / total visitors * 100)
        # (x100 to make it larger usually interactions per article << total visitors)
        # here minimum weight is 1 (because 2 ** 0 == 1), maximum weight is unbounded
        # weight increases exponentially with number of interactions because audience of two randomly sampled articles
        # may not be mutually exclusive
        visitors = time_period / datetime.timedelta(days=30) * monthly_visitors
        exposure = 2 ** (stats[key_metrics].sum(axis=1) / visitors * 100)

        return exposure

    # Input validation
    if num_topics <= 0:
        raise ValueError('num_topics must be a positive integer.')

    # Load news
    webhoseio = WebhoseIO(token=PRIVATE_API_KEY2)

    results = pd.DataFrame.from_records(
        webhoseio.query(
            EndPoint.ENRICHED_NEWS,
            filters={'language': 'english', 'category': 'politics', 'site.country': 'AU'},
            time_since=time_since,
            max_batches=1,
            output_dir=os.path.join(ROOT_DIR, 'datasets', 'news', 'recent-news-politics-au')
        )
    )

    sites = pd.DataFrame.from_records(results.pop('site'))
    articles = pd.DataFrame.from_records(results.pop('article'))

    site_traffics = pd.json_normalize(sites.pop('domain_ranks').to_list())
    page_social_media_stats = pd.json_normalize(articles.pop('social').to_list())

    articles = articles[['uuid', 'url', 'published', 'author', 'title', 'text', 'words']]
    articles['exposure'] = calculate_exposure(pd.concat([site_traffics, page_social_media_stats], axis=1),
                                              time_period=time_since)
    articles['text'] = articles['text'].apply(sent_tokenize)  # optimize this!
    articles = articles.explode('text')

    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    model = TFAutoModelForSequenceClassification.from_pretrained(
        os.path.join(ROOT_DIR, 'claim_detection', 'final-fine-tuned-models', 'distilroberta-base'),
        custom_objects={'tfa.metrics.F1Score': tfa.metrics.F1Score})

    sentences = tokenizer(articles['text'].to_list(), padding='max_length', truncation=True,
                          return_tensors='tf').data
    topics = model.predict(sentences)

    viewed_topics = (pd.DataFrame({'topic': topics, 'exposure': articles['exposure']})
                     .groupby('topic')
                     .sum()  # get total views for each topic
                     .sort_values(by='exposure', axis=0, ascending=False, kind='mergesort', ignore_index=True))
    # should exposure be linear? e.g. is a single article with 1 exposure just as good as two articles with 0.5?

    return viewed_topics[:num_topics]


def main():
    # Test only
    # df = get_au_political_news_since(time_since=datetime.timedelta(days=3), max_articles=100)
    hot_topics = get_hot_topics(10, time_since=datetime.timedelta(days=3))
    print(hot_topics)
    pass


if __name__ == '__main__':
    main()
