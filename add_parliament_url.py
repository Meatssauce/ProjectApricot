import pandas as pd
import numpy as np


# Load datasets about politicians
df = pd.read_csv('data/parliament-members.csv')

# Make suburl for www.aph.gov.au for each politician
mp = df['Role'].str.lower().str.contains('representative')
senator = df['Role'].str.lower().str.contains('senator')

if ~(mp ^ senator).any():
    raise ValueError('All roles must be either MP or Senator')

first_name_prefix = df['Name'].str.extract(r'^(\w)')[0].str.upper()
last_name = df['Name'].str.extract(r'.*\s(\w+)$')[0].str.capitalize()

df['Parliament url'] = np.where(mp & ~senator, first_name_prefix + '_' + last_name + '_MP', np.nan)
df['Parliament url'] = np.where(senator & ~mp, 'Senator_' + last_name, df['Parliament url'])
df['Parliament url'] = 'https://www.aph.gov.au/' + df['Parliament url']

df.to_csv('datasets/parliament-members.csv')



