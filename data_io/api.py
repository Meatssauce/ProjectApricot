import warnings
from enum import Enum
import datetime
import webhoseio


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


def to_timestamp(days_ago: int) -> int:
    proper_time = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    return int(proper_time.timestamp())


def overwrite_query():
    """Overwrite official query method of webhoseio with a more user-friendly alternative that also provides warnings
    and input parameters validation"""
    def wrapped_query(end_point_str, param_dict=None):
        if param_dict['keywords'] and 'category' in param_dict:
            warnings.warn('Using non-empty keywords with category may cause relevant results to be missing.',
                          ParamsWarning)
        if (sorting_policy := param_dict.get('sort')) and sorting_policy != SortingPolicy.get_default().value:
            warnings.warn('\nSorting by any sort parameter value other than CRAWL_DATE (default) may result in missing '
                          'important posts. \nIf data integrity is important, stick with the default recommended sort '
                          'parameter value of crawl date, and consume the data as a stream.', ParamsWarning)
        if end_point_str == DataType.NEWS_BLOGS_AND_FORMS.value and 'category' in param_dict:
            raise ParamsError("Cannot query NEWS_BLOGS_AND_FORMS with parameter 'category'.")

        query_message = ''
        if param_dict['keywords']:
            query_message += f"{param_dict['keywords']} "
        if "language" in param_dict:
            query_message += f"language:{param_dict['language']}"

        param_dict['q'] = query_message
        del param_dict['keywords'], param_dict['language']

        return func(end_point_str, param_dict)

    func = webhoseio.query
    webhoseio.query = wrapped_query
