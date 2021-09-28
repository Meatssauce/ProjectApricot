import pandas as pd
import numpy as np
from joblib import load, dump
import json
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# Load model and data
model = SentenceTransformer('all-mpnet-base-v2')  # SOTA for sentence embedding according to official docs
politicians = pd.read_csv('datasets/parliament-members.csv', index_col='Name')

# Compute tweet-based semantic embedding for each politician indexed by politician name
politician_embeddings = []
for handle in tqdm(politicians['Twitter Handle']):
    if handle is np.nan:
        politician_embeddings.append([])
    else:
        # Load tweets by the politician
        tweets_data = []
        with open(f'datasets/tweets/{handle}.json', 'r') as f:
            for line in f:
                tweets_data.append(json.loads(line))

        # Append empty embedding if tweets have no content todo: investigate reason for no content in tweets df
        if 'content' not in (tweets := pd.DataFrame.from_records(tweets_data)).columns:
            politician_embeddings.append([])
            continue

        # Remove urls and then hashtags from tweets
        tweets.content = tweets.content.str.replace(r'(https?://)?(www\.)?\w+(\.\w+)+(/\S*)?', '', regex=True)
        tweets.content = tweets.content.str.replace(r'#[A-Za-z0-9_]+', '', regex=True)

        # Produce a 1D embedding by taking the mean
        embeddings = model.encode(tweets.content.to_list())
        politician_embeddings.append(embeddings.mean(axis=0))
politician_embeddings = pd.DataFrame(politician_embeddings, index=politicians.index)

# Save embeddings
politician_embeddings.to_csv('datasets/tweet-based-embeddings.csv')
