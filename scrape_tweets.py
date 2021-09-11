import os

user_handle = 'SenatorRennick'

# Using OS library to call CLI commands in Python
os.system(f"snscrape --jsonl --max-results 100 twitter-search 'from:{user_handle}'> data/tweets/{user_handle}.json")
