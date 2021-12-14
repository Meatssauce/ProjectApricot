import itertools
import os
import warnings
from enum import Enum
import datetime
from math import ceil
from typing import Any

import joblib
import pandas as pd
import ujson
from tqdm import tqdm
from multiprocessing import Process, Queue, Lock
from collections.abc import Iterable, Iterator
import time
import requests
from functools import cache

from definitions import ROOT_DIR

PRIVATE_API_KEY = "12fede3e-49db-40a0-adc9-62d69ba00005"
PRIVATE_API_KEY2 = "92f9b97c-1976-4dd6-8ceb-90aac2f62272"


# class DummyWebhoseio:
#     def __init__(self):
#         self.results = iter(({'docs': [['dummy'] * 100], 'totalResults': 233},
#                              {'docs': [['dummy'] * 100], 'totalResults': 233},
#                              {'docs': [['dummy'] * 33], 'totalResults': 233}))
#
#     def config(self, token):
#         pass
#
#     def query(self, a, b):
#         return {'docs': [['dummy'] * 100], 'totalResults': 233}
#
#     def get_next(self):
#         return next(self.results)


class EndPoint(Enum):
    NEWS_BLOGS_AND_FORMS = "filterWebContent"
    ENRICHED_NEWS = "nseFilter"
    # REVIEWS = "reviewFilter"


# class Format(Enum):
#     JSON = 'json'
#     XML = 'xml'
#     RSS = 'rss'
#     EXCEL = 'excel'


class SortingPolicy(Enum):
    CRAWL_DATE = 'crawled'
    RELEVANCY = 'relevancy'
    PUBLISHED = 'published'
    THREAD_PUBLISHED = 'thread.published'
    FACEBOOK_LIKES = 'social.facebook.likes'
    FACEBOOK_SHARES = 'social.facebook.shares'
    FACEBOOK_COMMENTS = 'social.facebook.comments'
    GOOGLE_PLUS_SHARES = 'social.gplus.shares'
    PINTEREST_SHARES = 'social.pinterest.shares'
    LINKEDIN_SHARES = 'social.linkedin.shares'
    STUMBLEUPON_SHARES = 'social.stumbledupon.shares'
    VK_SHARES = 'social.vk.shares'
    REPLIES_COUNT = 'replies_count'
    PARTICIPANTS_COUNT = 'participants_count'
    SPAM_SCORE = 'spam_score'
    PERFORMANCE_SCORE = 'performance_score'
    DOMAIN_RANK = 'domain_rank'
    ORD_IN_THREAD = 'ord_in_thread'
    RATING = 'rating'
    UPDATED = 'updated'

    @classmethod
    def get_default(cls):
        """
            Returns the default sorting parameter of the API.

            Sorting by any sort parameter value other than CRAWL_DATE (default) may result in missing important posts.
            If data integrity is important, stick with the default recommended sort parameter value of crawl date, and
            consume the data as a stream.
        """
        return SortingPolicy.CRAWL_DATE


class ParamsWarning(Warning):
    pass


class ParamsError(ValueError):
    pass


class CoolDownIterator(Iterator):
    """Thread-safe iterator that with a blocking cool down after each iteration."""

    def __init__(self, iterable: Iterable, cool_down: datetime.timedelta = datetime.timedelta(seconds=0)):
        self._lock = Lock()
        self._cool_down = ceil(abs(cool_down).total_seconds())
        self._next_yield_time = float('-inf')
        self._iterable = iter(iterable) if iterable else itertools.count()

    def __next__(self):
        with self._lock:
            if time.perf_counter() < self._next_yield_time:
                time.sleep(self._next_yield_time - time.perf_counter())

            self._next_yield_time = time.perf_counter() + self._cool_down

            return self._iterable.__next__()

    def __iter__(self):
        return self


class WebhoseIO:
    def __init__(self, token=None):
        self._next_call = None
        self._session = requests.Session()
        self._token = token
        self._end_of_queue = 'DONE'

    def query(self,
              end_point: EndPoint,
              keywords: str = '',
              filters: dict = None,
              time_since: datetime.timedelta = datetime.timedelta(days=3),
              sort_: str = SortingPolicy.get_default().value,
              batch_size: int = 100,
              max_batches: int = float('inf'),
              cool_down_duration: datetime.timedelta = datetime.timedelta(seconds=1),
              output_dir: Any = None,
              multiprocess: bool = False,
              **kwargs) -> list:
        """
        A more user-friendly alternative of webhoseio's official query method that shows warnings and uses
        multithreading to maximise speed. See https://docs.webz.io/reference#news-blogs-discussions-api-overview for
        details on parameters and filters.

        :param end_point: type of data we want from the query
        :param keywords: input to search engine - e.g. ("Donald Trump" OR president) -Obama
        :param filters: special parameters used to filter search results
        :param time_since: time since earliest possible result
        :param sort_: metric by which search results are sorted
        :param batch_size: number of results returned each API call
        :param max_batches: maximum number of batches of results to receive from API (use a small value to
        limit API calls)
        :param cool_down_duration: amount of time between each successive API call (default - 1 second)
        :param output_dir: will save output to this directory if specified
        :param multiprocess: whether to call API in a parallel process
        :return: a list containing all the results without the metadata
        """
        # Show warnings
        warnings.warn("The webhose news API does not return warnings as expected. This method only provides minimal "
                      "input validation. \nPlease check dashboard on API website for error logs if you keep getting 0 "
                      "results.")
        if keywords and 'category' in filters:
            warnings.warn('Using non-empty keywords with category may cause relevant results to be missing.',
                          ParamsWarning)
        if sort_ != SortingPolicy.get_default().value:
            warnings.warn('\nSorting by any sort parameter value other than CRAWL_DATE (default) may result in missing'
                          ' important posts. \nIf data integrity is important, stick with the default recommended sort'
                          ' parameter value of crawl date, and consume the data as a stream.', ParamsWarning)
        if batch_size < 100:
            warnings.warn(f'Setting batch size to {batch_size} instead of the default 100 may lead to extra API '
                          f'calls.')
        # Basic input validation - incomplete
        if end_point == EndPoint.NEWS_BLOGS_AND_FORMS:
            if 'category:' in filters:
                raise ParamsError("Cannot query NEWS_BLOGS_AND_FORMS with parameter 'category'.")
            if 'site.country:' in filters or 'item.country:' in filters:
                raise ParamsError("Invalid country filter for querying NEWS_BLOGS_AND_FORMS.")
        elif end_point == EndPoint.ENRICHED_NEWS:
            if 'thread.country:' in filters or 'item.country:' in filters:
                raise ParamsError("Invalid country filter for querying ENRICHED_NEWS.")
        else:
            raise NotImplementedError()

        to_date = datetime.datetime.now()
        from_date = to_date - abs(time_since)

        # Parse query string
        query_string = ''
        if keywords:
            query_string += f'{keywords} '
        if filters:
            query_string += ' '.join(f'{k}:{v}' for k, v in filters.items())
        # Parse call parameters
        params = {
            'q': query_string,
            'ts': int(from_date.timestamp()),
            'sort': sort_,
            'size': batch_size,
            **kwargs
        }

        call_iterator = CoolDownIterator(itertools.count() if max_batches == float('inf') else range(max_batches),
                                         cool_down_duration)
        results_key = 'posts' if end_point == EndPoint.NEWS_BLOGS_AND_FORMS else 'docs'
        results = []

        # Get first page
        if response := self._query(end_point.value, params):
            results += response[results_key]

        # Get other pages
        if response and response['moreResultsAvailable'] > 0:
            total_calls = min(max_batches, ceil(response['totalResults'] / batch_size))

            if multiprocess:
                # Call API in another thread
                response_queue = Queue()

                writer_process = Process(target=self._response_writer, args=(response_queue, call_iterator))
                writer_process.start()

                for _ in tqdm(range(1, total_calls), initial=1, total=total_calls, unit='batch'):
                    if (response := response_queue.get()) == self._end_of_queue:
                        break
                    results += response[results_key]

                writer_process.join()
            else:
                for _ in tqdm(range(1, total_calls), initial=1, total=total_calls, unit='batch'):
                    next(call_iterator)
                    if not (response := self._get_next()):
                        break
                    results += response[results_key]

                    if response['moreResultsAvailable'] == 0:
                        break

        # Create a new folder under output_dir, then save results and parameters
        if output_dir:
            new_folder_name = f'{from_date.strftime("%e-%b-%Y")}-{to_date.strftime("%e-%b-%Y")}'
            output_dir = os.path.join(output_dir, new_folder_name)

            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                ujson.dump(results, f)
            with open(os.path.join(output_dir, 'params.joblib'), 'wb') as f:
                joblib.dump(params, f)

        return results

    def _get_next(self):
        return self._query(self._next_call[1:])

    def _query(self, end_point_str, param_dict=None):
        if param_dict is not None:
            param_dict.update({"token": self._token})
            param_dict.update({"format": "json"})

        response = self._session.get("http://webhose.io/" + end_point_str, params=param_dict)
        if response.status_code != 200:
            raise Exception(response.text)

        _output = response.json()
        self._next_call = _output['next']
        return _output

    def _response_writer(self, queue: Queue, call_iterator: CoolDownIterator):
        for _ in call_iterator:
            if not (response := self._get_next()):
                break
            queue.put(response)

            if response['moreResultsAvailable'] == 0:
                break
        queue.put(self._end_of_queue)


def main():
    # Get recent news about politics
    # webhoseio = WebhoseIO(token=PRIVATE_API_KEY2)

    # Test only
    # output = webhoseio.query(
    #     end_point=EndPoint.ENRICHED_NEWS,
    #     # keywords="\"Anne Webster\"",
    #     filters={
    #         'language': 'english',
    #         'category': 'politics',
    #         'site.country': 'AU',
    #         # 'thread.country': 'AU'
    #     },
    #     time_since=datetime.timedelta(days=15),
    #     batch_size=100,
    #     max_batches=3,
    # )
    # print(output)

    # webhoseio.query(
    #     EndPoint.ENRICHED_NEWS,
    #     filters={'language': 'english', 'category': 'politics', 'site.country': 'AU'},
    #     time_since=datetime.timedelta(days=30),
    #     max_batches=700,
    #     output_dir=os.path.join(ROOT_DIR, 'datasets', 'news', 'recent-news-politics-au')
    # )

    # Get historical news of politicians
    webhoseio = WebhoseIO(token=PRIVATE_API_KEY2)
    politicians = pd.read_csv(os.path.join(ROOT_DIR, 'datasets', 'parliament-members.csv'))['Name']

    politicians_news = []
    for name in tqdm(politicians):
        results = webhoseio.query(
            EndPoint.NEWS_BLOGS_AND_FORMS,
            keywords=f"\"{name}\"",
            filters={'language': 'english', 'thread.country': 'AU'},
            time_since=datetime.timedelta(days=30),
            max_batches=1,
        )

        if not results:
            continue

        news = pd.DataFrame.from_records(results)[['uuid', 'url', 'published', 'author', 'title', 'text']]
        news['politician'] = name

        politicians_news.append(news)
    politicians_news = pd.concat(politicians_news, axis=0, ignore_index=True)

    # remove none ascii characters
    politicians_news['text'] = politicians_news['text'].str.encode('ascii', 'ignore').str.decode('ascii')
    # standardise
    politicians_news['text'] = politicians_news['text'].str.replace(';', ',').str.replace(r'\\s+', r'\\s', regex=True)\
        .str.replace(r'\\n+', r'\\n', regex=True)

    politicians_news.to_csv(os.path.join(ROOT_DIR, 'datasets', 'news', 'au_parliament_members_news.csv'), index=False)


if __name__ == '__main__':
    main()
