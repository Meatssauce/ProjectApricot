import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# Load model and data
# current SOTA: all-mpnet-base-v2(best), all-MiniLM-L6-v2(a lot faster and still good performance)
model = SentenceTransformer('all-MiniLM-L6-v2')
politicians = pd.read_csv('../datasets/parliament-members.csv', index_col='Name')

# Compute tweet-based semantic embedding for each politician indexed by politician name, then semantic embedding of
# each tweet
politician_embeddings = []
tweet_embeddings = []
bad_handles = []
for name, handle in tqdm(zip(politicians.index, politicians['Twitter Handle']), total=len(politicians)):
    if handle is np.nan:
        politician_embeddings.append([])
        tweet_embeddings.append(pd.DataFrame())
    else:
        # Load tweets by the politician
        tweets_data = []
        with open(f'../datasets/tweets/{handle}.json', 'r') as f:
            for line in f:
                tweets_data.append(json.loads(line))

        # Append empty embedding if tweets have no content todo: investigate reason for no content in tweets df
        if 'content' not in (tweets := pd.DataFrame.from_records(tweets_data)).columns:
            politician_embeddings.append([])
            tweet_embeddings.append(pd.DataFrame())
            bad_handles.append(handle)
            continue

        # Remove urls and then hashtags from tweets
        tweets.content = tweets.content.str.replace(r'(https?://)?(www\.)?\w+(\.\w+)+(/\S*)?', '', regex=True)
        tweets.content = tweets.content.str.replace(r'#[A-Za-z0-9_]+', '', regex=True)

        # Compute semantic embedding for each tweet
        embeddings = model.encode(tweets.content.to_list())
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings['name'] = name
        tweet_embeddings.append(df_embeddings)

        # Produce a 1D embedding by taking the mean
        politician_embeddings.append(embeddings.mean(axis=0))
tweet_embeddings = pd.concat(tweet_embeddings, axis=0)
politician_embeddings = pd.DataFrame(politician_embeddings, index=politicians.index)

# Save embeddings
os.makedirs('../datasets', exist_ok=True)
tweet_embeddings.to_csv('../datasets/tweet-embeddings.csv', index=False)
politician_embeddings.to_csv('../datasets/tweet-based-embeddings.csv')
print(f'error triggers {bad_handles}')
# error triggers ['joshwilsonmp', 'SteveIronsMP', 'JohnAlexanderMP', 'lidia__thorpe', 'lukejgosling', 'SHendersonMP',
# 'stephenjonesalp', 'JamesMcGrathLNP']
# they are all empty json files
