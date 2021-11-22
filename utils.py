import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Union, Tuple, Set

import pandas as pd
import requests


class VoteType(Enum):
    # html class for each vote type on theyvoteforyou.org
    VeryStronglyFor = 'position-voted-very-strongly-for'
    StronglyFor = 'position-voted-strongly-for'
    ModeratelyFor = 'position-voted-moderately-for'
    Mixed = 'position-voted-a-mixture-of-for-and-against'
    ModeratelyAgainst = 'position-voted-moderately-against'
    StronglyAgainst = 'position-voted-strongly-against'
    VeryStronglyAgainst = 'position-voted-very-strongly-against'


def make_set():
    return set()


@dataclass
class Party:
    name: str
    members: set = field(default_factory=make_set, init=False)
    start_date: datetime = None
    end_date: datetime = None

    def add(self, member) -> None:
        member.party = self
        self.members.add(member)


@dataclass
class Politician:
    name: str
    # affiliation_history: list
    electorate: str
    votes: list
    friends: set
    rebellion_rate: float = None
    attendance_rate: float = None
    party: Party = None


def get_news_about(search_terms: Union[List[str], Set[str], Tuple[str], str], from_: datetime = None,
                   to: datetime = None) -> pd.DataFrame:
    """Get scraped news about specified keywords using newsapi.com"""
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    from_msg = '' if not from_ else f"from={str(from_)}&"
    to_msg = '' if not to else f"to={str(to)}&"

    results = []
    for search_term in search_terms:
        url = ('https://newsapi.org/v2/everything?'
               f'q={search_term}&'
               f'{from_msg}'
               f'{to_msg}'
               'sortBy=popularity&'
               'sources=bbc-news&'
               'apiKey=7cdfcc7872594e14b197bafdb9b6da04')
        response = requests.get(url)
        results.append(pd.DataFrame.from_dict(response.json()['articles']))
    results = pd.concat(results, axis=0, ignore_index=True)

    return results


def get_federal_parliament_members() -> List[str]:
    """Get a dataframe containing names of current members of the federal parliament"""
    return pd.read_csv(os.path.join('datasets', 'parliament-members.csv'))['Name'].to_list()
