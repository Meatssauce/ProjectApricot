# Scraping news with newsapi
# See documentation at https://newsapi.org/docs/get-started#search
import os

import pandas as pd
import requests

keywords = "Elon Musk"

# Newsapi
url = ('https://newsapi.org/v2/everything?'
       f'q={keywords}'
       '&from=2021-11-10'
       '&sortBy=popularity'
       '&apiKey=7cdfcc7872594e14b197bafdb9b6da04')

response = requests.get(url)
df = pd.DataFrame.from_dict(response.json()['articles'])


# Mediastack
# url = ("http://api.mediastack.com/v1/news"
#        "?access_key=952d94143a37e6002f9336f9b10e4ba4"
#        "&languages=en"
#        f"&keywords={keywords}"
#        '&categories=-sports'
#        '&sort=published_desc'
#        '&limit=10'
#        )
#
# headers = {
#     'x-rapidapi-host': "mediastack.p.rapidapi.com",
#     'x-rapidapi-key': "d372caac62mshe79e42568ad4d83p1e0e8ejsn3d85308ba3ba"
#     }
#
# response = requests.request("GET", url, headers=headers)
# df = pd.DataFrame.from_dict(response.json()['data'])


df.to_csv(os.path.join('..', 'datasets', 'news', f'{keywords}.csv'), index=False)
print(response.json())
