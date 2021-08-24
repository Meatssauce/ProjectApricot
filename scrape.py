import logging
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
from parameters import sources, request_headers
import re


# Get get parsed html from a url
def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=request_headers)
    soup = BeautifulSoup(response.content, 'html5lib')
    return soup


def get_postcodes() -> list:
    post_codes = []
    seed_url = sources['pa'] + 'state-postcodes/'
    states = ['act', 'nsw', 'nt', 'qld', 'sa', 'tas', 'vic', 'wa']

    logging.debug(f'Starting to scrape post codes from f{len(states)} states')

    for state in tqdm(states):
        url = seed_url + state
        soup = get_soup(url)
        list_ = soup.find('ul', {'class': 'pclist'})
        if not list_ and soup.find('div', {'id': 'content'}).find('h1').text == 'Error':
            raise ValueError('Failed to find list on the page. Check if url is correct.')

        post_codes = [item.find('a').text for item in list_.findAll('li', recursive=False)]

    logging.debug(post_codes)

    return post_codes


def get_politicians() -> dict:
    """
    Scrapes all Australian politicians according to Wikipedia.

    :return: The name and party of each politician
    """

    def _scrape_table_of_politicians(url: str) -> tuple:
        members, parties = [], []

        soup = get_soup(url)
        table = soup.find('table')
        if 'Members_of_the_Tasmanian_Legislative_Council' in url:
            table = table.findNext('table')
        rows = table.findAll('tr')
        # headers = [re.sub('\n+', '', header.text) for header in rows[0].findAll('th')]
        #
        # assert politicians_data.keys() == headers

        for row in rows[1:]:
            elements = row.findAll('td')
            sort_value = elements[0].find('span')
            if sort_value:
                member = elements[0].find('span')['data-sort-value']
                member = re.match(r'^(([a-zA-Z\'-]*,*\s*)?[a-zA-z\'-]+)', member)[0]
                member = ' '.join(reversed(member.split(', ')))
            else:
                member = elements[0].text
            member = re.sub(r'\s*(\[.*\]|\d)$', '', member)
            member = re.sub(r'^(Hon|Dr|Hon Dr)\s*', '', member)
            members.append(member)

            if len(elements) == 5 or len(elements) == 6:
                party = elements[2].find('a').text
                # politicians_data['electorate'].append(elements[3].find('a').text)
                # politicians_data['end_term'].append(elements[4].text)
                # politicians_data['years_in_office'].append(elements[5].text)
            else:
                party = elements[1].text
            party = re.sub(r'\s*(\[.*\]|\d)$', '', party)
            parties.append(party)
                # politicians_data['electorate'].append(elements[2].text)

        return members, parties

    politicians_data = {'member': [], 'party': []}

    seed_url = sources['politicians']
    hostname = re.findall(r'^(http[s]?://[^/]+/)', seed_url)[0]
    soup = get_soup(seed_url)
    main_content = soup.find('div', {'class': 'mw-parser-output'})
    sublists = main_content.findAll('ul', recursive=False)
    to_visit = []
    for sublist in sublists[:2]:
        items = sublist.findAll('li', recursive=False)
        for item in items:
            to_visit.append(hostname + item.find('a')['href'])

    visited = set()
    for url in tqdm(to_visit):
        if url in visited:
            continue

        members, parties = _scrape_table_of_politicians(url)
        politicians_data['member'] += members
        politicians_data['party'] += parties

    return politicians_data


# Tests
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # post_codes = get_postcodes()
    politicans_data = get_politicians()
    df = pd.DataFrame.from_dict(politicans_data)
    df.to_csv('data/politicians_data.csv', index=False)
