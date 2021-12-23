from typing import Union

import requests
import pandas as pd
import datetime
from tqdm import tqdm
import numpy as np
import ujson
import os

from data_io import CoolDownIterator

PRIVATE_API_KEY = "r3oO02PHBbjQQl1+jAHh"
API_URL = "https://theyvoteforyou.org.au/api/v1/"


def get_all_people(return_df=False) -> Union[dict, pd.DataFrame, None]:
    session = requests.Session()
    response = session.get(API_URL + 'people.json', params={'key': PRIVATE_API_KEY})
    if response.status_code != 200:
        return
    if return_df:
        people = pd.DataFrame.from_records(response.json())
        details = pd.json_normalize(people['latest_member']).rename(columns={'id': 'latest_member.id'})
        return pd.concat([people['id'], details], axis=1)
    return response.json()


def get_person(id_: int) -> Union[dict, None]:
    session = requests.Session()
    response = session.get(API_URL + f'people/{id_}.json', params={'key': PRIVATE_API_KEY})
    if response.status_code != 200:
        return
    return response.json()


def get_all_policies() -> Union[dict, None]:
    session = requests.Session()
    response = session.get(API_URL + 'policies.json', params={'key': PRIVATE_API_KEY})
    if response.status_code != 200:
        return
    return response.json()


def get_policy(id_: int) -> Union[dict, None]:
    session = requests.Session()
    response = session.get(API_URL + f'policies/{id_}.json', params={'key': PRIVATE_API_KEY})
    if response.status_code != 200:
        return
    return response.json()


def get_all_divisions() -> Union[dict, None]:
    session = requests.Session()
    response = session.get(API_URL + 'divisions.json', params={'key': PRIVATE_API_KEY})
    if response.status_code != 200:
        return
    return response.json()


def get_division(id_: int) -> Union[dict, None]:
    session = requests.Session()
    response = session.get(API_URL + f'divisions/{id_}.json', params={'key': PRIVATE_API_KEY})
    if response.status_code != 200:
        return
    return response.json()


def get_all_personal_details():
    people = get_all_people()
    all_details = [get_person(person['id']) for person in CoolDownIterator(tqdm(people), datetime.timedelta(seconds=1))]
    return all_details


data = get_all_people()
...