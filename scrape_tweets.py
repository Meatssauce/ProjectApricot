import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from facebook_scraper import get_posts

# Parameters
max_tweets = 100

# Load politician dataset
df = pd.read_csv('data/au_parliament_members_data.csv')

# Scrape tweets
twitter_handles = df['Twitter'].str.extract(r'^.+/(\w+)$')[0]
for handle in tqdm(twitter_handles.values):
    if handle is not np.nan:
        os.system(f"snscrape --jsonl --max-results {max_tweets} --since 2010-01-01 twitter-search 'from:{handle}' "
                  f"> datapo/tweets/{handle}.json")

# user_handle = 'SenatorRennick'

# Scrape facebook posts
# facebook_urls = df['Facebook']
# for url in tqdm(facebook_urls.values):
#     if url is not np.nan:
#         os.system(f"snscrape --jsonl --max-results {max_tweets} --since 2010-01-01 twitter-search 'from:{handle}' "
#                   f"> dataset/tweets/{handle}.json")
