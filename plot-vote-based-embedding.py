import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load, dump
from category_encoders import OrdinalEncoder
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from os import makedirs
from umap import UMAP

politicians_of_interest = ['Scott Morrison', 'Simon Birmingham', 'Peter Dutton', 'Linda Burney', 'Rachel Siewert']
policies_of_interest = ['Increasing surveillance powers',
                        'Protecting citizens  privacy',
                        'Protecting whistleblowers',
                        'Continuing Detention Orders  CDOs',
                        'More scrutiny of intelligence services   police',
                        'Increasing consumer protections',
                        'A Free Trade Agreement with China',
                        'Greater public scrutiny of the Trans Pacific Partnership negotiations',
                        'A same sex marriage plebiscite',
                        'Civil celebrants having the right to refuse to marry same sex couples']


def make_vote_based_scatter_plot(data, x, y, hue, kind, remark=''):
    if remark:
        remark = '-' + remark

    # Create simple scatter plot with data point labels
    # sns.scatterplot(data=voting_records, x='Component 1', y='Component 2', hue=stances)
    sns.scatterplot(data=data, x=x, y=y)
    for name in politicians_of_interest:
        if name in data.index:
            plt.text(data.loc[name, x], data.loc[name, y], name)
    plt.savefig(f'plots/vote-based-scatter-plot-simple{remark}.png')
    plt.close()

    # Create join scatter plot
    sns.jointplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        kind=kind  # or 'kde' or 'hex'
    )
    plt.savefig(f'plots/vote-based-scatter-plot-joint{remark}.png')
    plt.close()


def make_voter_based_scatter_plot(data, x, y, remark=''):
    if remark:
        remark = '-' + remark

    # Create scatter plot for reduced dimensions
    # sns.scatterplot(data=voting_records, x='Component 1', y='Component 2', hue=parties)
    sns.scatterplot(data=data, x=x, y=y)
    for i in policies_of_interest:
        if i in data.index:
            plt.text(data.loc[i, x], data.loc[i, y], i)
    plt.savefig(f'plots/voter-based-scatter-plot-for-policies{remark}.png')
    plt.close()


makedirs('plots', exist_ok=True)

#
# Scatter plot of politicians
#

# Produce vote-based embeddings for politicians from policy voting records
df = pd.read_csv('datasets/parliament-policies-voting-records.csv')
voting_records = df.drop('Policy', axis=1).pivot(index='Politician', columns='URL')

# Encode vote type as ordinal variables
voting_records = voting_records.fillna(-1)
mapping = [{'col': col, 'mapping': {
    'VeryStronglyAgainst': 0, 'StronglyAgainst': 1, 'ModeratelyAgainst': 2, 'Mixed': 3, -1: 3,
    'ModeratelyFor': 4, 'StronglyFor': 5, 'VeryStronglyFor': 6}} for col in voting_records.columns]
encoder = OrdinalEncoder(return_df=True, mapping=mapping)
voting_records = encoder.fit_transform(voting_records)

# Get party of each politician and pick Liberals' voting records
df = pd.read_csv('datasets/parliament-members.csv', index_col='Name')
parties = df.loc[[name for name in voting_records.index], 'Party']
# voting_records = voting_records[parties == 'Liberal Party']
top_parties = parties.value_counts().index[:4]
parties.loc[~parties.isin(top_parties)] = "Other"

# # Uncomment this block to show mark left vs right or authoritarian vs libertarian politicians
# # Hue vector for stance on key issues
# stances = np.where(voting_records['Type', 'https://theyvoteforyou.org.au/policies/44'] > 3, 1, 0)

# Reduce to two dimensions with PCA
pca = PCA(2)
voting_records_2d = pd.DataFrame(pca.fit_transform(voting_records), index=voting_records.index,
                                 columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?
voting_records_2d *= -1  # for cosmetic purposes

# Plot
make_vote_based_scatter_plot(data=voting_records_2d, x='Component 1', y='Component 2', hue=parties, kind='scatter',
                             remark='pca')

# Reduce to two dimensions with UMAP
umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
voting_records_2d = pd.DataFrame(umap.fit_transform(voting_records), index=voting_records.index,
                                 columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?
voting_records_2d *= -1  # for cosmetic purposes

# Plot
make_vote_based_scatter_plot(data=voting_records_2d, x='Component 1', y='Component 2', hue=parties, kind='scatter',
                             remark='umap')

# =========================================

#
# Scatter plot of policies
#

# Produce voter-based embeddings for policies from policy voting records
df = pd.read_csv('datasets/parliament-policies-voting-records.csv')
voting_records = df.drop('URL', axis=1).pivot(index='Policy', columns='Politician')

# Encode vote type as ordinal variables
voting_records = voting_records.fillna(-1)
mapping = [{'col': col, 'mapping': {
    'VeryStronglyAgainst': 0, 'StronglyAgainst': 1, 'ModeratelyAgainst': 2, 'Mixed': 3, -1: 3,
    'ModeratelyFor': 4, 'StronglyFor': 5, 'VeryStronglyFor': 6}} for col in voting_records.columns]
encoder = OrdinalEncoder(return_df=True, mapping=mapping)
voting_records = encoder.fit_transform(voting_records)

# Get party of each politician and pick Liberals' voting records
# df = pd.read_csv('datasets/parliament-members.csv', index_col='Name')
# parties = df.loc[[name for name in voting_records.index], 'Party']
# # voting_records = voting_records[parties == 'Liberal Party']
# top_parties = parties.value_counts().index[:4]
# parties.loc[~parties.isin(top_parties)] = "Other"

# Reduce to two dimensions with PCA
pca = PCA(2)
voting_records_2d = pd.DataFrame(pca.fit_transform(voting_records), index=voting_records.index,
                                 columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?

# Plot
make_voter_based_scatter_plot(data=voting_records_2d, x='Component 1', y='Component 2', remark='pca')

# Reduce to two dimensions with UMAP
umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
voting_records_2d = pd.DataFrame(umap.fit_transform(voting_records), index=voting_records.index,
                                 columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?

# Plot
make_voter_based_scatter_plot(data=voting_records_2d, x='Component 1', y='Component 2', remark='umap')
