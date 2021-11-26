import warnings
from enum import Enum
import datetime
from math import ceil
import webhoseio
from tqdm import tqdm
from multiprocessing import Process, Queue
from collections import Iterator
from threading import Lock
import time

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


class Format(Enum):
    JSON = 'json'
    XML = 'xml'
    RSS = 'rss'
    EXCEL = 'excel'


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


def _query_writer(queue: Queue, call_api, end_point_str: str, param_dict: dict):
    result_batch = call_api(end_point_str, param_dict)
    api_rate_limiter = RateLimiter(param_dict['cool_down_duration'])

    batch_size = param_dict['size']
    initial_batch_size = len(result_batch['docs'])
    expected_total_results = min(param_dict['max_results'], result_batch['totalResults'])

    for results_so_far in tqdm(range(initial_batch_size, expected_total_results + 1, batch_size),
                               initial=initial_batch_size, total=expected_total_results,
                               unit='article', unit_scale=batch_size):
        queue.put(result_batch)

        if results_so_far + batch_size > expected_total_results + 1:
            break
        else:
            next(api_rate_limiter)
            result_batch = webhoseio.get_next()
    queue.put('DONE')


def _wrapped_query(end_point_str: str, param_dict=None):
    """A more user-friendly alternative of webhoseio's official query method that shows warnings and uses
    multithreading to maximise speed."""
    # Basic input validation - not complete
    if end_point_str == DataType.NEWS_BLOGS_AND_FORMS.value:
        if 'category:' in param_dict['q']:
            raise ParamsError("Cannot query NEWS_BLOGS_AND_FORMS with parameter 'category'.")
        if 'site.country:' in param_dict['q'] or 'item.country:' in param_dict['q']:
            raise ParamsError("Invalid country filter for querying NEWS_BLOGS_AND_FORMS.")
    elif end_point_str == DataType.ENRICHED_NEWS.value:
        if 'thread.country:' in param_dict['q'] or 'item.country:' in param_dict['q']:
            raise ParamsError("Invalid country filter for querying ENRICHED_NEWS.")
    elif end_point_str == DataType.REVIEWS.value:
        if 'thread.country:' in param_dict['q'] or 'site.country:' in param_dict['q']:
            raise ParamsError("Invalid country filter for querying ENRICHED_NEWS.")
    warnings.warn("\nThe webhose news API does not return warnings as expected. This method only provides minimal "
                  "input validation. \nPlease check dashboard on API website for error logs if you keep getting 0 "
                  "results.")

    results_queue = Queue()
    docs = []

    writer_process = Process(target=_query_writer, args=(results_queue, _query, end_point_str, param_dict))
    writer_process.start()

    # Process api call results
    while True:
        results = results_queue.get()
        if results == 'DONE':
            break
        else:
            docs += results['docs']
    writer_process.join()

    return docs


# def _overwrite_query():
#     """Overwrite official query method of webhoseio with a more user-friendly alternative that also provides warnings
#     and input parameters validation."""
#
#
#     query = webhoseio.query
#     webhoseio.query = _wrapped_query


def get_params(keywords: str = '',
               filters: dict = None,
               ts: int = None,
               sort_: str = SortingPolicy.get_default().value,
               size: int = 10,
               max_results: int = float('inf'),
               cool_down_duration: datetime.timedelta = datetime.timedelta(seconds=1),
               format_: str = Format.JSON,
               warnings_: bool = False) -> dict:
    """
    Get parameters for API query.

    :param keywords: Input to search engine - e.g. ("Donald Trump" OR president) -Obama
    :param filters: dictionary containing filters - see webhoseio documentation for usage
    :param ts: time of earliest result in unix time (13 digits)
    :param sort_: how results are sorted - Note: Sorting by any sort parameter value other than
    crawl date (default) may result in missing important posts. If data integrity is important,
    stick with the default recommended sort parameter value of crawl date, and consume the data as a stream.
    :param size: number of articles returned per batch
    :param max_results: maximum number of results to receive from API (use a small value to limit API calls)
    :param cool_down_duration: amount of time between each API call (default - 1 second)
    :param format_: output result in this format - integer between 2-100 inclusive
    :param warnings_: show warnings via a warnings object in output
    :return: dictionary containing valid parameters for the API
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

    # Parse query string
    query_string = ''
    if keywords:
        query_string += f'{keywords} '
    if filters:
        query_string += parse_filters(filters)

    # Set default time since to 3 lookback_window ago
    if not ts:
        ts = to_timestamp(days_ago=3)

    return {
        'q': query_string,
        'ts': ts,
        'sort': sort_,
        'size': size,
        'max_results': max_results,
        'cool_down_duration': cool_down_duration,
        'format': format_,
        'warnings': warnings_,
    }


def to_timestamp(days_ago: int) -> int:
    proper_time = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    return int(proper_time.timestamp())


# webhoseio = DummyWebhoseio()

# Overwrite webhoseio.query with wrapper
if webhoseio.query is not _wrapped_query:
    _query = webhoseio.query
    webhoseio.query = _wrapped_query


def main():
    # Test only
    # import webhoseio

    webhoseio.config(token=PRIVATE_API_KEY)
    query_params = get_params(
        # keywords="\"Mark Dreyfus\"",
        filters={
            'language': 'english',
            'category': 'politics',
            'site.country': 'AU'
        },
        ts=to_timestamp(days_ago=3),
        size=100,
        # max_results=50,
        warnings_=True,
    )

    output = webhoseio.query(DataType.ENRICHED_NEWS.value, query_params)
    print(output)
    # print(output['posts'][0]['text'])  # Print the text of the first post
    # print(output['posts'][0]['published'])  # Print the text of the first post publication date
    #
    # # Get the next batch of posts
    # output = webhoseio.get_next()
    # if output['posts']:
    #     print(output['posts'][0]['thread']['site'])  # Print the site of the first post
    # else:
    #     print(output)


if __name__ == '__main__':
    main()
