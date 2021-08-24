import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Parameters
from tqdm import tqdm

request_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip',
    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
    'Upgrade-Insecure-Requests': '1',
    # 'Referer': 'https://www.google.com/'
}
sources = {'tvfy': 'https://theyvoteforyou.org.au/', 'pa': 'https://postcodes-australia.com/'}


# Get get parsed html from a url
def get_soup(url):
    response = requests.get(url, headers=request_headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup


def get_postcodes():
    post_codes = []
    root_url = sources['pa'] + 'state-postcodes/'
    states = ['act', 'nsw', 'nt', 'qld', 'sa', 'tas', 'vic', 'wa']

    logging.debug(f'Starting to scrape post codes from f{len(states)} states')

    for state in tqdm(states):
        url = root_url + state
        soup = get_soup(url)
        list_ = soup.find('ul', {'class': 'pclist'})
        if not list_:
            raise ValueError('Failed to find list on the page. Check if url is correct.')

        for item in list_.findAll('li', recursive=False):
            code = item.find('a').text
            post_codes.append(code)

    logging.debug(post_codes)

    return post_codes


logging.basicConfig(level=logging.INFO)
post_codes = get_postcodes()
