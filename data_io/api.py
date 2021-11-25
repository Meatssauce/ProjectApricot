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


def _overwrite_query():
    """Overwrite official query method of webhoseio with a more user-friendly alternative that also provides warnings
    and input parameters validation."""

    def wrapped_query(end_point_str, param_dict=None):
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

        warnings.warn("The webhose news API does not return warnings as expected. This method only provides minimal "
                      "input validation. Please check dashboard on API website for error logs if you keep getting 0 "
                      "results.")

        return func(end_point_str, param_dict)

    func = webhoseio.query
    webhoseio.query = wrapped_query


def get_params(keywords: str = '', filters: dict = None, ts: int = None,
               sort_: str = SortingPolicy.get_default().value, format_: str = Format.JSON,
               warnings_: bool = False) -> dict:
    """
    Get parameters for API query.

    :param keywords: Input to search engine - e.g. ("Donald Trump" OR president) -Obama
    :param filters: dictionary containing filters - see webhoseio documentation for usage
    :param ts: time of earliest result in unix time (13 digits)
    :param sort_: how results are sorted - Note: Sorting by any sort parameter value other than
    crawl date (default) may result in missing important posts. If data integrity is important,
    stick with the default recommended sort parameter value of crawl date, and consume the data as a stream.
    :param format_: output result in this format
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

    # Set default time since to 3 days ago
    if not ts:
        ts = to_timestamp(days_ago=3)

    return {
        'q': query_string,
        'sort': sort_,
        'ts': ts,
        'format': format_,
        'warnings': warnings_
    }


_overwrite_query()


def main():
    # Test only
    import webhoseio

    webhoseio.config(token="12fede3e-49db-40a0-adc9-62d69ba00005")
    query_params = get_params(
        keywords="\"Mark Dreyfus\"",
        filters={
            'language': 'english',
            'category': 'politics',
            'site.country': 'AU'
        },
        ts=to_timestamp(days_ago=3),
        warnings_=True
    )

    output = webhoseio.query(DataType.NEWS_BLOGS_AND_FORMS.value, query_params)
    print(output['posts'][0]['text'])  # Print the text of the first post
    print(output['posts'][0]['published'])  # Print the text of the first post publication date

    # Get the next batch of posts
    output = webhoseio.get_next()
    if output['posts']:
        print(output['posts'][0]['thread']['site'])  # Print the site of the first post
    else:
        print(output)


if __name__ == '__main__':
    main()

