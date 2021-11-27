import warnings
from enum import Enum
import datetime
from math import ceil
from tqdm import tqdm
from multiprocessing import Process, Queue
from collections import Iterator
from threading import Lock
import time
import requests

PRIVATE_API_KEY = "12fede3e-49db-40a0-adc9-62d69ba00005"


class DummyWebhoseio:
    def __init__(self):
        self.results = iter(({'docs': [['dummy'] * 100], 'totalResults': 233},
                             {'docs': [['dummy'] * 100], 'totalResults': 233},
                             {'docs': [['dummy'] * 33], 'totalResults': 233}))

    def config(self, token):
        pass

    def query(self, a, b):
        return {'docs': [['dummy'] * 100], 'totalResults': 233}

    def get_next(self):
        return next(self.results)


class DataType(Enum):
    NEWS_BLOGS_AND_FORMS = "filterWebContent"
    ENRICHED_NEWS = "nseFilter"
    REVIEWS = "reviewFilter"


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


class RateLimiter(Iterator):
    """Iterator that yields a value at most once every 'interval' seconds."""

    def __init__(self, interval: datetime.timedelta = datetime.timedelta(seconds=0)):
        self.lock = Lock()
        self.interval = ceil(interval.total_seconds())
        self.next_yield_time = time.perf_counter() + self.interval

    def __next__(self):
        with self.lock:
            if time.perf_counter() < self.next_yield_time:
                time.sleep(self.next_yield_time - time.perf_counter())
            self.next_yield_time = time.perf_counter() + self.interval


class WebhoseIO:
    def __init__(self, token=None):
        self.next_call = None
        self.session = requests.Session()
        self.token = token
        self.max_batches = None
        self.cool_down_duration = None
        self.params = None
        self._end_of_queue = 'DONE'

    def set_params(self,
                   keywords: str = '',
                   filters: dict = None,
                   time_since: datetime.timedelta = datetime.timedelta(days=3),
                   sort_: str = SortingPolicy.get_default().value,
                   batch_size: int = 100,
                   max_batches: int = float('inf'),
                   cool_down_duration: datetime.timedelta = datetime.timedelta(seconds=1),
                   **kwargs) -> None:
        """
        Get parameters for API query. See https://docs.webz.io/reference#news-blogs-discussions-api-overview for
        details on parameters and filters.

        :param keywords: input to search engine - e.g. ("Donald Trump" OR president) -Obama
        :param filters: speical parameters used to filter search results
        :param time_since: time since earliest possible result
        :param sort_: metric by which search results are sorted
        :param batch_size: number of results returned each API call
        :param max_batches: maximum number of batches of results to receive from API (use a small value to
        limit API calls)
        :param cool_down_duration: amount of time between each successive API call (default - 1 second)
        :return: None
        """

        def parse_filters(filters: dict) -> str:
            substrings = [f'{k}:{v}' for k, v in filters.items()]
            return ' '.join(substrings)

        # Show warnings
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

        # Parse query string
        query_string = ''
        if keywords:
            query_string += f'{keywords} '
        if filters:
            query_string += parse_filters(filters)

        self.max_batches = max_batches
        self.cool_down_duration = cool_down_duration
        self.params = {
            'q': query_string,
            'time_since': int((datetime.datetime.now() - abs(time_since)).timestamp()),
            'sort': sort_,
            'size': batch_size,
            **kwargs
        }

    def query(self, end_point_str: str) -> list:
        """A more user-friendly alternative of webhoseio's official query method that shows warnings and uses
        multithreading to maximise speed."""
        # Basic input validation - not complete
        if end_point_str == DataType.NEWS_BLOGS_AND_FORMS.value:
            if 'category:' in self.params['q']:
                raise ParamsError("Cannot query NEWS_BLOGS_AND_FORMS with parameter 'category'.")
            if 'site.country:' in self.params['q'] or 'item.country:' in self.params['q']:
                raise ParamsError("Invalid country filter for querying NEWS_BLOGS_AND_FORMS.")
        elif end_point_str == DataType.ENRICHED_NEWS.value:
            if 'thread.country:' in self.params['q'] or 'item.country:' in self.params['q']:
                raise ParamsError("Invalid country filter for querying ENRICHED_NEWS.")
        elif end_point_str == DataType.REVIEWS.value:
            if 'thread.country:' in self.params['q'] or 'site.country:' in self.params['q']:
                raise ParamsError("Invalid country filter for querying ENRICHED_NEWS.")
        warnings.warn("\nThe webhose news API does not return warnings as expected. This method only provides minimal "
                      "input validation. \nPlease check dashboard on API website for error logs if you keep getting 0 "
                      "results.")

        response_queue = Queue()

        # Write API response to queue in another thread
        writer_process = Process(target=self._response_writer, args=(response_queue, end_point_str))
        writer_process.start()

        # Process API responses added to the queue in main thread
        if (response := response_queue.get()) == self._end_of_queue:
            return []

        docs = response['docs']
        batch_size = self.params['size']
        num_batches_to_read = min(self.max_batches, ceil(response['totalResults'] / batch_size))

        for _ in tqdm(range(1, num_batches_to_read), initial=1, total=num_batches_to_read, unit='batch'):
            if (response := response_queue.get()) == self._end_of_queue:
                break
            else:
                docs += response['docs']

        writer_process.join()

        return docs

    def _get_next(self):
        return self._query(self.next_call[1:])

    def _query(self, end_point_str, param_dict=None):
        if param_dict is not None:
            param_dict.update({"token": self.token})
            param_dict.update({"format": "json"})

        response = self.session.get("http://webhose.io/" + end_point_str, params=param_dict)
        if response.status_code != 200:
            raise Exception(response.text)

        _output = response.json()
        self.next_call = _output['next']
        return _output

    def _response_writer(self, queue: Queue, end_point_str: str):
        response = None
        api_rate_limiter = RateLimiter(self.cool_down_duration)
        i = 0

        while True:
            response = self._get_next() if response else self._query(end_point_str, self.params)
            queue.put(response)

            if i >= self.max_batches - 1 or response['moreResultsAvailable'] == 0:
                break
            else:
                next(api_rate_limiter)
                i += 1
        queue.put(self._end_of_queue)


# def to_timestamp(days_ago: int) -> int:
#     proper_time = datetime.datetime.now() - datetime.timedelta(days=days_ago)
#     return int(proper_time.timestamp())


def main():
    # Test only
    webhoseio = WebhoseIO(token=PRIVATE_API_KEY)

    webhoseio.set_params(
        # keywords="\"Mark Dreyfus\"",
        filters={
            'language': 'english',
            'category': 'politics',
            'site.country': 'AU'
        },
        time_since=datetime.timedelta(days=3),
        batch_size=100,
        max_batches=3,
    )
    output = webhoseio.query(DataType.ENRICHED_NEWS.value)
    print(output)
    # print(output['posts'][0]['text'])  # Print the text of the first post
    # print(output['posts'][0]['published'])  # Print the text of the first post publication date
    #
    # # Get the next batch of posts
    # output = webhoseio._get_next()
    # if output['posts']:
    #     print(output['posts'][0]['thread']['site'])  # Print the site of the first post
    # else:
    #     print(output)


if __name__ == '__main__':
    main()
