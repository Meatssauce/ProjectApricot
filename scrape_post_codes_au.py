import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
from parameters import *


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
