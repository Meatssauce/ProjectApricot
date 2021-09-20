import logging
import random

import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re

from typing import Tuple, List, Dict
from collections import namedtuple, defaultdict
from joblib import load, dump

from utils import Politician, VoteType

# Parameters
request_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip',
    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
    'Upgrade-Insecure-Requests': '1',
    # 'Referer': 'https://www.google.com/'
}


# Get get parsed html from a url
def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=request_headers)
    return BeautifulSoup(response.content, 'html5lib')


# Get hostname aka root url from a url
def get_root(url: str) -> str:
    return re.findall(r'^(http[s]?://[^/]+)', url)[0]


# Remove commas and quotation marks from string
def cleaned(string: str) -> str:
    string = re.sub(r'\W', ' ', string)  # remove symbols and extra white space
    return re.sub(r'(^\s+|\s+$)', '', string)  # remove white space at beginning and end


def scrape_postcodes_au() -> list:
    post_codes = []
    seed_url = 'https://postcodes-australia.com/state-postcodes/'
    states = ['act', 'nsw', 'nt', 'qld', 'sa', 'tas', 'vic', 'wa']

    logging.debug(f'Starting to scrape post codes from f{len(states)} states')

    for state in tqdm(states):
        url = seed_url + state
        soup = get_soup(url)
        list_ = soup.find('ul', class_='pclist')
        if not list_ and soup.find('div', {'id': 'content'}).find('h1').text == 'Error':
            raise ValueError('Failed to find list on the page. Check if url is correct.')

        post_codes = [item.find('a').text for item in list_.findAll('li', recursive=False)]

    logging.debug(post_codes)

    return post_codes


def scrape_all_parliment_members() -> dict:
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
                member = elements[0].find('span')['dataset-sort-value']
                member = re.match(r'^(([a-zA-Z\'-]*,*\s*)?[a-zA-z\'-]+)', member)[0]
                member = ' '.join(reversed(member.split(', ')))
            else:
                member = elements[0].text
            member = cleaned(member)
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

    logging.info('Starting to scrape politicians from Wikipedia')

    politicians_data = {'member': [], 'party': []}

    seed_url = 'https://en.wikipedia.org/wiki/List_of_Australian_politicians'
    hostname = get_root(seed_url)
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


# def scrape_policy_voting_histories(soup: BeautifulSoup, url: str) -> List[dict]:
#     # Scrape politician's voting history
#     policies = soup.find('ul', class_='policy-comparision-list')
#     policies = [(url + policy.a['href'], sanitised(policy.text))
#                 for policy in policies.findAll('li', recursive=False)]
#     df_policies = pd.DataFrame(policies, columns=['URL', 'Description'])
#     return df_policies


def scrape_politician_info(url: str, politicians_data: defaultdict, voting_history_data: defaultdict,
                           matrix: defaultdict, shortcut=False) -> None:
    def scrape_policy_url_from_intermediary(url: str) -> str:
        """Scrapes absolute url to policy from the politician policy details page."""
        title = get_soup(url).find('h1', class_='media-heading long-title')
        return get_root(url) + title.findAll('a')[1]['href']

    page = get_soup(url)

    # Scrape info about politician
    summary = page.find('div', class_='media-body')
    politician_name = cleaned(summary.find('h1').find('span').text)
    # todo: find regex that does preserves O'Bryan and Foo-Zoo but not O', 'O, Foo- or -Foo
    politicians_data['Name'].append(politician_name)
    politicians_data['Party'].append(cleaned(summary.find('span', class_='org').text))
    politicians_data['Role'].append(cleaned(summary.find('span', class_='title').text))
    politicians_data['Electorate'].append(cleaned(summary.find('span', class_='electorate').text))
    try:
        rebellion_rate = summary.find('span', class_='member-rebellions').text
        if 'never' not in rebellion_rate.lower():
            rebellion_rate = re.search(r'\d+\.?\d*', rebellion_rate)[0]
            rebellion_rate = float(rebellion_rate) / 100
        else:
            rebellion_rate = 0.
        politicians_data['Rebellion'].append(rebellion_rate)

        attendance_rate = summary.find('span', class_='member-attendance').text
        attendance_rate = re.search(r'\d+\.?\d*', attendance_rate)[0]
        attendance_rate = float(attendance_rate) / 100
        politicians_data['Attendance'].append(attendance_rate)
    except AttributeError:
        politicians_data['Rebellion'].append('')
        politicians_data['Attendance'].append(np.nan)

    # Scrape info about policy
    for type_ in VoteType:
        section = page.find('div', class_=type_.value)
        if section:
            list_ = section.find('ul')
            for item in list_.findAll('li'):
                voting_history_data['Politician'].append(cleaned(politician_name))
                voting_history_data['Type'].append(type_.name)
                voting_history_data['Policy'].append(cleaned(item.text))

                root_url = get_root(url)
                sub_url = item.a['href']
                if shortcut:
                    # todo: check if full policy url always == get_root(url) + last two suffix of item.a['href']
                    policy_url = root_url + re.search(r'.*(/policies/\d+)\n?$', sub_url)[1]
                else:
                    policy_url = scrape_policy_url_from_intermediary(root_url + sub_url)
                voting_history_data['URL'].append(policy_url)

    # Scrape info about politician's friends
    page = get_soup(url + '/friends')
    table = page.find('table')
    for row in table.findAll('tr')[1:]:
        cells = row.findAll('td')
        friend_name = cleaned(cells[1].text)
        agreement = re.match(r'\d+\.?\d*', cells[0].text)[0]
        agreement = float(agreement) / 100

        matrix[politician_name][friend_name] = agreement
        matrix[friend_name][politician_name] = agreement


def scrape_australian_parliament_members(shortcut=False) -> Tuple[pd.DataFrame, pd.DataFrame, defaultdict]:
    logging.info('Starting to scrape politicians from TheyVoteForYou')

    # Scrape url of all parliament members
    seed_url = 'https://theyvoteforyou.org.au/people'
    hostname = get_root(seed_url)
    page = get_soup(seed_url)
    list_ = page.find('div', class_='container main-content').find('ol')
    to_visit = []
    for item in list_.findAll('li', recursive=False):
        to_visit.append(hostname + item.find('a')['href'])
    random.shuffle(to_visit)  # shuffle for anti-crawler detection

    # Scrape info of all parliament members
    politicians_data = defaultdict(list)
    voting_history_data = defaultdict(list)
    friendship_matrix = defaultdict(defaultdict)
    visited = set()
    for url in tqdm(to_visit):
        if url in visited:
            continue
        visited.add(url)

        scrape_politician_info(url, politicians_data, voting_history_data, friendship_matrix, shortcut=True)
    df_politicians = pd.DataFrame.from_dict(politicians_data)
    df_voting_history = pd.DataFrame.from_dict(voting_history_data)

    return df_politicians, df_voting_history, friendship_matrix


def scrape_policies(urls: List[str]) -> pd.DataFrame:
    data = {'Name': [], 'Description': [], 'URL': []}
    for url in urls:
        page = get_soup(url)
        header = page.find('div', class_='page-header')
        data['Name'].append(cleaned(header.find('h1', class_='long-title').text))
        data['Description'].append(cleaned(header.find('div', class_='policytext').text))
        data['URL'].append(url)
    return pd.DataFrame.from_dict(data)


def scrape_politician_donations():
    """Scrape info about a parliament member's donors and donations"""
    pass


def scrape_australian_party_donations():
    """Scrape info about a party's donors and donations"""
    pass


def find_politician_stakes():
    """Scrape info about how much stake a politician holds and in what area"""
    pass


def find_party_stakes():
    """Scrape info about how much stake a party holds and in what area"""
    pass


# Tests
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Scrape all parliament members
    # politicans_data = scrape_all_parliment_members()
    # df = pd.DataFrame.from_dict(politicans_data)
    # df.to_csv('dataset/australian_and_state_parliament_members.csv', index=False)

    # Scrape info about politicians, their policy voting history and friends
    df_politicians, df_voting_history, friendship_matrix = scrape_australian_parliament_members(shortcut=False)

    df_politicians.to_csv('dataset/au_parliament_members_data.csv', index=False)
    df_voting_history.to_csv('dataset/au_parliament_policies_voting_data.csv', index=False)
    with open('data/friendship_matrix.joblib', 'wb') as f:
        dump(friendship_matrix, f)

    # Scrape info about policies
    # df_voting_history = pd.read_csv('dataset/au_parliament_policies_voting_data.csv')
    # df_policies = scrape_policies(df_voting_history['URL'].tolist())
    # df_policies.to_csv('dataset/policies.csv', index=False)
