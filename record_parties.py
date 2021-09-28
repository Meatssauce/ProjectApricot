from joblib import load, dump
import pandas as pd
import numpy as np
from utils import Politician, Party

df = pd.read_csv('data/parliament-members.csv')

all_parties = set()
for party_name in df['Party'].unique():
    Party(party_name)


    all_parties.add()
