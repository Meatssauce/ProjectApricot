import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load, dump
from category_encoders import OrdinalEncoder
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA

#
# Cluster plot of politicians
#

# Produce vote-based embeddings for politicians from policy voting records
df = pd.read_csv('dataset/au_parliament_policies_voting_data.csv')
voting_records = df.drop('Policy', axis=1).pivot(index='Politician', columns='URL')

# Encode vote type as ordinal variables
voting_records = voting_records.fillna(-1)
mapping = [{'col': col, 'mapping': {
    'VeryStronglyAgainst': 0, 'StronglyAgainst': 1, 'ModeratelyAgainst': 2, 'Mixed': 3, -1: 3,
    'ModeratelyFor': 4, 'StronglyFor': 5, 'VeryStronglyFor': 6}} for col in voting_records.columns]
encoder = OrdinalEncoder(return_df=True, mapping=mapping)
voting_records = encoder.fit_transform(voting_records)

# Get party of each politician and pick Liberals' voting records
df = pd.read_csv('dataset/au_parliament_members_data.csv', index_col='Name')
parties = df.loc[[name for name in voting_records.index], 'Party']
# voting_records = voting_records[parties == 'Liberal Party']
top_parties = parties.value_counts().index[:4]
parties.loc[~parties.isin(top_parties)] = "Other"

# Hue vector for stance on key issues
stances = np.where(voting_records['Type', 'https://theyvoteforyou.org.au/policies/44'] > 3, 1, 0)

# Reduce to two dimensions
pca = PCA(2)
voting_records = pd.DataFrame(pca.fit_transform(voting_records), index=voting_records.index,
                              columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?
voting_records *= -1  # for cosmetic purposes

# Create scatter plot for reduced dimensions
# sns.scatterplot(data=voting_records, x='Component 1', y='Component 2', hue=stances)
# sns.scatterplot(data=voting_records, x='Component 1', y='Component 2')
sns.jointplot(
    data=voting_records,
    x='Component 1',
    y='Component 2',
    hue=parties,
    kind='scatter'  # or 'kde' or 'hex'
)
prominent_politicians = ['Scott Morrison', 'Simon Birmingham', 'Peter Dutton', 'Linda Burney', 'Rachel Siewert']
for name in prominent_politicians:
    if name in voting_records.index:
        plt.text(voting_records.loc[name, 'Component 1'], voting_records.loc[name, 'Component 2'], name)
plt.show()

# =========================================

#
# Cluster plot of policies
#

# Produce voter-based embeddings for policies from policy voting records
df = pd.read_csv('dataset/au_parliament_policies_voting_data.csv')
voting_records = df.drop('URL', axis=1).pivot(index='Policy', columns='Politician')

# Encode vote type as ordinal variables
voting_records = voting_records.fillna(-1)
mapping = [{'col': col, 'mapping': {
    'VeryStronglyAgainst': 0, 'StronglyAgainst': 1, 'ModeratelyAgainst': 2, 'Mixed': 3, -1: 3,
    'ModeratelyFor': 4, 'StronglyFor': 5, 'VeryStronglyFor': 6}} for col in voting_records.columns]
encoder = OrdinalEncoder(return_df=True, mapping=mapping)
voting_records = encoder.fit_transform(voting_records)

# Get party of each politician and pick Liberals' voting records
# df = pd.read_csv('dataset/au_parliament_members_data.csv', index_col='Name')
# parties = df.loc[[name for name in voting_records.index], 'Party']
# # voting_records = voting_records[parties == 'Liberal Party']
# top_parties = parties.value_counts().index[:4]
# parties.loc[~parties.isin(top_parties)] = "Other"

# Reduce to two dimensions
pca = PCA(2)
voting_records = pd.DataFrame(pca.fit_transform(voting_records), index=voting_records.index,
                              columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?

# Create scatter plot for reduced dimensions
# sns.scatterplot(data=voting_records, x='Component 1', y='Component 2', hue=parties)
sns.scatterplot(data=voting_records, x='Component 1', y='Component 2')
prominent_policies = voting_records.index[20:30]
for name in prominent_policies:
    if name in voting_records.index:
        plt.text(voting_records.loc[name, 'Component 1'], voting_records.loc[name, 'Component 2'], name)

plt.show()
