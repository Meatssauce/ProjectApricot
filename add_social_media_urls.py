import pandas as pd
import numpy as np
import requests
import re

from joblib import dump
from tqdm import tqdm

from scrape import get_soup, cleaned

# Load politician info
df = pd.read_csv('data/au_parliament_members_data.csv')

# Find url to search result for each politician
search_urls = df['Name'].str.replace(r'\s+', '+', regex=True)
search_urls = 'https://www.aph.gov.au/Senators_and_Members/Parliamentarian_Search_Results?q=' + search_urls

# Scrape social media links from top result, flag names with no results
error = [False] * len(search_urls)
facebook_urls, twitter_urls = [np.nan] * len(search_urls), [np.nan] * len(search_urls)
for i, (url, name) in enumerate(zip(tqdm(search_urls), df['Name'].values)):
    try:
        soup = get_soup(url)
        results = soup.find('div', class_='search-filter-results search-filter-results-snm row')
        top_result = results.find('div', class_='row border-bottom padding-top')

        if name not in cleaned(top_result.find('h4', class_='title').text):
            raise AttributeError()
        if facebook := top_result.find('a', class_='social facebook'):
            facebook_urls[i] = facebook['href']
        if twitter := top_result.find('a', class_='social twitter margin-right'):
            twitter_urls[i] = twitter['href']
    except AttributeError:
        error[i] = True

df['Facebook'] = facebook_urls
df['Twitter'] = twitter_urls
df['Error'] = error
df.to_csv('data/au_parliament_members_data.csv', index=False)
