import logging
import random

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re


# Parameters
from utils import Politician

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
    soup = BeautifulSoup(response.content, 'html5lib')
    return soup


# Get hostname aka root url from a url
def get_root(url: str) -> str:
    return re.findall(r'^(http[s]?://[^/]+/)', url)[0]


# Remove commas and quotation marks from string
def sanitised(string: str) -> str:
    return re.sub(r'[,\"\']+', '', string)


def scrape_postcodes_au() -> list:
    post_codes = []
    seed_url = 'https://postcodes-australia.com/state-postcodes/'
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
                member = elements[0].find('span')['data-sort-value']
                member = re.match(r'^(([a-zA-Z\'-]*,*\s*)?[a-zA-z\'-]+)', member)[0]
                member = ' '.join(reversed(member.split(', ')))
            else:
                member = elements[0].text
            member = sanitised(member)
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


def scrape_australian_parliament_members() -> dict:
    def _scrape_politician_info(url: str, politicians_data: dict) -> dict:
        soup = get_soup(url)
        summary = soup.find('div', class_='media-body')

        politicians_data['name'].append(sanitised(summary.find('h1').find('span').text))
        politicians_data['party'].append(sanitised(summary.find('span', class_='org').text))
        politicians_data['role'].append(sanitised(summary.find('span', class_='title').text))
        politicians_data['electorate'].append(sanitised(summary.find('span', class_='electorate').text))
        try:
            politicians_data['rebellion'].append(sanitised(summary.find('span', class_='member-rebellions').text))
            politicians_data['attendance'].append(sanitised(summary.find('span', class_='member-attendance').text))
        except AttributeError:
            politicians_data['rebellion'].append('')
            politicians_data['attendance'].append('')

        issues = soup.find('ul', class_='policy-comparision-list')
        policies = [(issue.a['href'], sanitised(issue.text)) for issue in issues.findAll('li', recursive=False)]
        politicians_data['policies'].append(policies)

        friends_page = get_soup(url + '/friends')
        friends_table = friends_page.find('table')

        friends = []
        for row in friends_table.findAll('tr', recursive=False):
            cells = row.findAll('td')
            friends.append((sanitised(cell.text) for cell in cells))
        politicians_data['friends'].append(friends)

        return politicians_data

    logging.info('Starting to scrape politicians from TheyVoteForYou')

    politicians_data = {'name': [], 'party': [], 'role': [], 'electorate': [], 'rebellion': [], 'attendance': [],
                        'policies': [], 'friends': []}

    seed_url = 'https://theyvoteforyou.org.au/people'
    hostname = get_root(seed_url)
    soup = get_soup(seed_url)
    list_ = soup.find('div', {'class': 'container main-content'}).find('ol')
    to_visit = []
    for item in list_.findAll('li', recursive=False):
        to_visit.append(hostname + item.find('a')['href'])
    random.shuffle(to_visit)  # shuffle for anti-crawler detection

    visited = set()
    for url in tqdm(to_visit):
        if url in visited:
            continue

        politicians_data = _scrape_politician_info(url, politicians_data)

    return politicians_data


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
    # post_codes = get_postcodes()
    # politicans_data = scrape_all_parliment_members()
    # df = pd.DataFrame.from_dict(politicans_data)
    # df.to_csv('data/australian_and_state_parliament_members.csv', index=False)

    politicans_data = scrape_australian_parliament_members()
    df = pd.DataFrame.from_dict(politicans_data)
    df.to_csv('data/australian_parliament_members_data.csv', index=False)
