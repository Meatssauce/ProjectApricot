import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from joblib import load, dump
from category_encoders import OrdinalEncoder
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from os import makedirs

makedirs('plots', exist_ok=True)

#
# Cluster plot of politicians based on their tweets
#

# Load data and use name as index
politicians = pd.read_csv('datasets/parliament-members.csv', index_col='Name')
embeddings = pd.read_csv('datasets/tweet-based-embeddings.csv', index_col='Name')
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

# Create simple scatter plot with data point labels
sns.scatterplot(data=politicians, x='Component 1', y='Component 2')
prominent_politicians = ['Scott Morrison', 'Simon Birmingham', 'Peter Dutton', 'Linda Burney', 'Rachel Siewert']
for name in prominent_politicians:
    if name in politicians.index:
        plt.text(politicians.loc[name, 'Component 1'], politicians.loc[name, 'Component 2'], name)
plt.savefig('plots/tweet-based-scatter-plot-simple.png')
plt.close()

# Create join scatter plot
sns.jointplot(
    data=politicians,
    x='Component 1',
    y='Component 2',
    hue='Party',
    kind='scatter'  # or 'kde' or 'hex'
)
plt.savefig('plots/tweet-based-scatter-plot-join.png')
plt.close()
