import re
from typing import Union
import requests
import pandas as pd
import datetime
from tqdm import tqdm
from data_io import CoolDownIterator
from functools import cache

PRIVATE_API_KEY = "r3oO02PHBbjQQl1+jAHh"


class TvfyData:
    def __init__(self, api_key: str, timeout: int = 300):
        self.API_KEY = api_key
        self.timeout = timeout
        self.session = requests.Session()

        self.URL_API = "https://theyvoteforyou.org.au/api/v1/"

    @staticmethod
    def _http_error_msg(response):
        return u'%s Client Error: %s for url: %s' % (response.status_code, response.reason, response.url)

    # @staticmethod
    # def _extract_latest_member(latest_member: pd.Series, drop_member_id: bool = False):
    #     details = pd.json_normalize(latest_member)
    #     if drop_member_id:
    #         return details.drop(columns=['id'])
    #     return details.rename(columns={'id': 'latest_member.id'})

    @cache
    def get_all_people(self) -> Union[dict, pd.DataFrame]:
        response = self.session.get(f'{self.URL_API}people.json', params={'key': self.API_KEY})

        if response.status_code != 200:
            raise requests.HTTPError(self._http_error_msg(response), response=response)
        return response.json()

    @cache
    def get_person(self, id_: int) -> dict:
        response = self.session.get(f'{self.URL_API}people/{id_}.json', params={'key': self.API_KEY})
        if response.status_code != 200:
            raise requests.HTTPError(self._http_error_msg(response), response=response)
        return response.json()

    @cache
    def get_all_policies(self) -> Union[dict, pd.DataFrame]:
        response = self.session.get(f'{self.URL_API}policies.json', params={'key': self.API_KEY})

        if response.status_code != 200:
            raise requests.HTTPError(self._http_error_msg(response), response=response)
        return response.json()

    @cache
    def get_policy(self, id_: int) -> dict:
        response = self.session.get(f'{self.URL_API}policies/{id_}.json', params={'key': self.API_KEY})
        if response.status_code != 200:
            raise requests.HTTPError(self._http_error_msg(response), response=response)
        return response.json()

    @cache
    def get_all_divisions(self) -> dict:
        response = self.session.get(f'{self.URL_API}divisions.json', params={'key': self.API_KEY})
        if response.status_code != 200:
            raise requests.HTTPError(self._http_error_msg(response), response=response)
        return response.json()

    @cache
    def get_division(self, id_: int) -> dict:
        response = self.session.get(f'{self.URL_API}divisions/{id_}.json', params={'key': self.API_KEY})
        if response.status_code != 200:
            raise requests.HTTPError(self._http_error_msg(response), response=response)
        return response.json()

    @cache
    def get_all_personal_details(self, cool_down: int = 0) -> Union[list, pd.DataFrame]:
        cool_down = datetime.timedelta(seconds=cool_down)
        all_details = [self.get_person(person['id'])
                       for person in CoolDownIterator(tqdm(self.get_all_people(), unit='person'), cool_down)]
        return all_details


client = TvfyData(PRIVATE_API_KEY)
people = pd.json_normalize(client.get_all_people())
policies = pd.json_normalize(client.get_all_policies())
all_personal_details = pd.json_normalize(client.get_all_personal_details())
...
