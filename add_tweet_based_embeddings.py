import pandas as pd
import numpy as np
from joblib import load, dump
import json
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# Load model and data
model = SentenceTransformer('all-mpnet-base-v2')  # SOTA for sentence embedding according to official docs
politicians = pd.read_csv('dataset/au_parliament_members_data.csv')

# Compute tweet-based semantic embedding for each politician
politician_embeddings = []
for name, handle in tqdm(zip(politicians['Name'], politicians['Twitter Handle']), total=len(politicians)):
    if handle is np.nan:
        politician_embeddings.append(np.nan)

    else:
        # Load tweets by the politician
        tweets_data = []
        with open(f'dataset/tweets/{handle}.json', 'r') as f:
            for line in f:
                tweets_data.append(json.loads(line))
        if 'content' not in (tweets := pd.DataFrame.from_records(tweets_data)).columns:
            politician_embeddings.append(np.nan)
            continue

        # Remove urls and then hashtags from tweets
        tweets.content = tweets.content.str.replace(r'(https?://)?(www\.)?\w+(\.\w+)+(/\S*)?', '', regex=True)
        tweets.content = tweets.content.str.replace(r'#[A-Za-z0-9_]+', '', regex=True)

        # Produce a 1D embedding by taking the mean
        embeddings = model.encode(tweets.content.to_list())
        politician_embeddings.append(embeddings.mean(axis=0))
politicians['embedding'] = politician_embeddings

politicians.to_csv('dataset/au_parliament_members_data.csv', index=False)
