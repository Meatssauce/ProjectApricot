import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
import seaborn as sns
from sklearn.decomposition import PCA
from os import makedirs
from umap import UMAP


def make_scatter_plots(data, x, y, hue, kind, remark=''):
    if remark:
        remark = '-' + remark

    # Create simple scatter plot with data point labels
    sns.scatterplot(data=data, x=x, y=y)
    prominent_politicians = ['Scott Morrison', 'Simon Birmingham', 'Peter Dutton', 'Linda Burney', 'Rachel Siewert']
    for name in prominent_politicians:
        if name in data.index:
            plt.text(data.loc[name, x], data.loc[name, y], name)
    plt.savefig(f'plots/tweet-based-scatter-plot-simple{remark}.png')
    plt.close()

    # Create join scatter plot
    sns.jointplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        kind=kind  # or 'kde' or 'hex'
    )
    plt.savefig(f'plots/tweet-based-scatter-plot-join{remark}.png')
    plt.close()


makedirs('plots', exist_ok=True)

# #
# # Scatter plot of politicians based on their tweets
# #
#
# # Load data and use name as index
# politicians = pd.read_csv('datasets/parliament-members.csv', index_col='Name')
# embeddings = pd.read_csv('datasets/tweet-based-embeddings.csv', index_col='Name')
# # politicians = pd.concat([politicians, embeddings], axis=1)
#
# # Keep only politicians with a tweet-based embedding
# politicians = politicians[~embeddings.iloc[:, 0].isna()]
# embeddings = embeddings[~embeddings.iloc[:, 0].isna()]
#
# # Replace niche parties with 'Other'
# # voting_records = voting_records[politicians['Party'] == 'Liberal Party']
# top_parties = politicians['Party'].value_counts().index[:4]
# politicians['Party'].loc[~politicians['Party'].isin(top_parties)] = "Other"
#
# # Reduce to two dimensions with PCA
# pca = PCA(2)
# lower_embeddings = pd.DataFrame(pca.fit_transform(embeddings), index=embeddings.index,
#                                 columns=['Component 1', 'Component 2'])
# data = pd.concat([politicians, lower_embeddings], axis=1)
#
# make_scatter_plots(data=data,
#                    x='Component 1',
#                    y='Component 2',
#                    hue='Party',
#                    kind='scatter',  # or 'kde' or 'hex'
#                    remark='pca'
#                    )
#
# # Reduce to two dimensions with UMAP
# umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
# lower_embeddings = pd.DataFrame(umap.fit_transform(embeddings), index=embeddings.index,
#                                 columns=['Component 1', 'Component 2'])
# data = pd.concat([politicians, lower_embeddings], axis=1)
#
# make_scatter_plots(data=data,
#                    x='Component 1',
#                    y='Component 2',
#                    hue='Party',
#                    kind='scatter',  # or 'kde' or 'hex'
#                    remark='umap'
#                    )

# ==========================================

#
# Scatter plot of tweets - topic modeling
#

# Load data and use name as index
embeddings = pd.read_csv('datasets/tweet-embeddings.csv')
del embeddings['name']

# Reduce to two dimensions with UMAP
print('Begin UMAP dimensionality reduction')
data = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine').fit_transform(embeddings)
print(f'{data.shape}')
cluster = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(data)
print('Begin UMAP dimensionality reduction')
data = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
print(f'{data.shape}')
data = pd.DataFrame(data, columns=['x', 'y'])
data['labels'] = cluster.labels_

fig, ax = plt.subplots(figsize=(20, 10))
outliers = data.loc[data['labels'] == -1, :]
clustered = data.loc[data['labels'] != -1, :]
plt.scatter(outliers['x'], outliers['y'], color='#BDBDBD', s=0.05)
plt.scatter(clustered['x'], clustered['y'], c=clustered['labels'], s=0.05, cmap='hsv_r')
plt.colorbar()
plt.savefig('plots/tweet-based-topic-modeling.png')
plt.close()
