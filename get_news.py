import json
import time
from typing import List, Tuple
import os
import requests
import pandas as pd
import re
from joblib import dump


def find_news(headers: dict, params: dict) -> Tuple[dict, list]:
    """Search news using newscatcher api. 10,000 calls, 1 API call/second, 1 month old articles"""
    # GET call api with the parameters provided
    response = requests.get("https://api.newscatcherapi.com/v2/search", headers=headers, params=params)
    if response.status_code != 200:
        raise APIError(f'API call failed')

    # Encode results and add search parameters to each result to be able to explore afterwards
    results = json.loads(response.text.encode())
    if 'articles' in results:
        for i in results['articles']:
            i['used_params'] = str(params)

    return results, results['articles']


def find_all_news(headers: dict, params:dict, max_articles: int = None) -> Tuple[List[dict], List[dict]]:
    """Extract news on every page"""
    all_results = []
    all_articles = []
    params['page'] = 1

    # Infinite loop which ends when all articles are extracted
    while True:
        time.sleep(1)  # wait 1 second between each call due to api limit
        print(f'Proceed extracting page number => {params["page"]}')

        try:
            results, articles = find_news(headers, params)
        except APIError as e:
            raise APIError(f'API call failed for page number => {params["page"]}') from e
        else:
            print(f'Done for page number => {params["page"]}')

            all_results.append(results)
            all_articles += articles
            params['page'] += 1

            if (max_articles and len(all_articles) >= max_articles) or params['page'] >= results['total_pages']:
                print("All articles have been extracted")
                break

    print(f'Number of extracted articles => {len(all_articles)}')

    return all_results, all_articles


class APIError(Exception):
    pass


def get_salient_regions(text: str, key_term: str) -> List[List[int]] or None:
    key_term = re.sub(r'\s+', ' ', key_term)
    regions = None
    offset = 30

    while result := re.search(rf'{key_term}', text, re.IGNORECASE):
        start, end = max(result.start() - offset, 0), min(result.end() + offset, len(text))
        if not regions:
            regions = [[start, end]]
        else:
            if start < regions[-1][1] < end:
                regions[-1][1] = end
            else:
                regions.append([start, end])

        text = text[result.end():]

    return regions


# Driver code
keywords = ' '
headers = {
    'x-api-key': "63US199aYQka9-nwR6xNtpH3WIEYAPLRuUijRWOxVwk"
    }
params = {
    "q": f'{keywords}',  # search terms, supports boolean operations, use inner double quotes for exact match
    "lang": "en",
    'countries': 'au',
    'to_rank': 10000,  # up to 10000th most popular result
    "sort_by": "relevancy",
    'page_size': 100,
    "page": 1,  # page number, not number of pages,
}

# Get single news
# results, articles = find_news(headers, params)
# pandas_table = pd.DataFrame(results['articles'])

# Get all news
results, articles = find_all_news(headers, params, max_articles=10)

# Save all articles
save_dir = os.path.join('datasets', 'news')
os.makedirs(save_dir, exist_ok=True)
articles_data = {k: [] for k in articles[0]}
for article in articles:
    for k, v in article.items():
        articles_data[k].append(v)
df = pd.DataFrame.from_dict(articles_data)
df.to_csv(os.path.join(save_dir, f'{keywords}.csv'), index=False),
# with open(os.path.join('news', 'results.joblib'), 'wb') as f:
#     dump(results, f)
print(articles)
