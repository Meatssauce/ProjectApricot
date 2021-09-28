import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load, dump
from category_encoders import OrdinalEncoder
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA

#
# Cluster plot of politicians based on their tweets
#

# Load data and use name as index
politicians = pd.read_csv('dataset/au_parliament_members_data.csv', index_col='Name')
embeddings = pd.read_csv('dataset/tweet_based_embeddings.csv', index_col='Name')
politicians = pd.concat([politicians, embeddings], axis=1)

# Keep only politicians with a tweet-based embedding
politicians = politicians[~embeddings.iloc[:, 0].isna()]
embeddings = embeddings[~embeddings.iloc[:, 0].isna()]

# Replace niche parties with 'Other'
# voting_records = voting_records[politicians['Party'] == 'Liberal Party']
top_parties = politicians['Party'].value_counts().index[:4]
politicians['Party'].loc[~politicians['Party'].isin(top_parties)] = "Other"

# Reduce to two dimensions
pca = PCA(2)
embeddings = pd.DataFrame(pca.fit_transform(embeddings), index=embeddings.index, columns=['Component 1', 'Component 2'])
politicians = pd.concat([politicians, embeddings], axis=1)

# Create scatter plot for reduced dimensions
# sns.scatterplot(data=voting_records, x='Component 1', y='Component 2', hue=stances)
# sns.scatterplot(data=voting_records, x='Component 1', y='Component 2')
sns.jointplot(
    data=politicians,
    x='Component 1',
    y='Component 2',
    hue='Party',
    kind='scatter'  # or 'kde' or 'hex'
)
prominent_politicians = ['Scott Morrison', 'Simon Birmingham', 'Peter Dutton', 'Linda Burney', 'Rachel Siewert']
for name in prominent_politicians:
    if name in politicians.index:
        plt.text(politicians.loc[name, 'Component 1'], politicians.loc[name, 'Component 2'], name)
plt.show()
