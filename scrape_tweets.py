import os
import warnings

import pandas as pd
from tqdm import tqdm
import numpy as np
from facebook_scraper import get_posts
import json
from requests.exceptions import HTTPError

# Parameters
max_tweets = 800

# Load politician dataset
df = pd.read_csv('dataset/au_parliament_members_data.csv')

# Scrape tweets
os.makedirs('dataset/tweets', exist_ok=True)
for handle, name in tqdm(zip(df['Twitter Handle'], df['Name']), total=len(df)):
    if handle is not np.nan:
        try:
            os.system(f"snscrape --jsonl --max-results {max_tweets} --since 2010-01-01 twitter-search 'from:{handle}' "
                      f"> dataset/tweets/{handle}.json")
        except UnicodeEncodeError:
            pass

# # Scrape facebook posts
# warnings.filterwarnings("ignore", category=UserWarning)
# os.makedirs('dataset/facebook_posts', exist_ok=True)
# for handle, name in tqdm(zip(df['Facebook Handle'], df['Name']), total=len(df)):
#     if handle is not np.nan:
#         try:
#             posts = []
#             for post in get_posts(handle, page_limit=10, cookies='assets/fb_cookie.json',
#                                   options={'progress': False, 'locale': 'en_AU'}):
#                 post['time'] = str(post['time'])
#                 post['shared_time'] = str(post['shared_time'])
#                 posts.append(post)
#
#             with open(f'dataset/facebook_posts/{handle}.json', 'w') as f:
#                 json.dump(posts, f)
#         except Exception:
#             pass
# warnings.filterwarnings("default", category=UserWarning)
