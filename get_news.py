import json
import time
from typing import List

import requests
import pandas as pd


def get_news(headers: dict, params: dict) -> dict:
    """Search news using newscatcher api. 10,000 calls, 1 API call/second, 1 month old articles"""
    # GET call api with the parameters provided
    response = requests.get("https://api.newscatcherapi.com/v2/search", headers=headers, params=params)
    if response.status_code != 200:
        raise APIError(f'API call failed')

    # Encode results and add search parameters to each result to be able to explore afterwards
    results = json.loads(response.text.encode())
    for i in results['articles']:
        i['used_params'] = str(params)

    return results


def get_all_news(headers: dict, params:dict) -> List[dict]:
    """Extract news on every page"""
    all_news_articles = []
    params['page'] = 1

    # Infinite loop which ends when all articles are extracted
    while True:
        time.sleep(1)  # wait 1 second between each call due to api limit
        print(f'Proceed extracting page number => {params["page"]}')

        try:
            results = get_news(headers, params)
        except APIError as e:
            raise APIError(f'API call failed for page number => {params["page"]}') from e
        else:
            print(f'Done for page number => {params["page"]}')

            all_news_articles += results['articles']
            params['page'] += 1

            if params['page'] > results['total_pages']:
                print("All articles have been extracted")
                break

    print(f'Number of extracted articles => {len(all_news_articles)}')

    return all_news_articles


class APIError(Exception):
    pass


# Driver code
headers = {
    'x-api-key': "63US199aYQka9-nwR6xNtpH3WIEYAPLRuUijRWOxVwk"
    }
params = {
    "q": "Elon Musk",  # search terms, supports boolean operations
    "lang": "en",
    'to_rank': 10000,  # up to 10000th most popular result
    'page_size': 100,
    "sort_by": "relevancy",
    "page": 1,  # page number, not number of pages
}
results = get_news(headers, params)
pandas_table = pd.DataFrame(results['articles'])
