import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load, dump
from category_encoders import OrdinalEncoder
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA

df = pd.read_csv('dataset/au_parliament_policies_voting_data.csv')
voting_records = df.drop('Policy', axis=1).pivot(index='Politician', columns='URL')

# Encode vote type as ordinal variables
voting_records = voting_records.fillna(-1)
mapping = [{'col': col, 'mapping': {
    'VeryStronglyAgainst': 0, 'StronglyAgainst': 1, 'ModeratelyAgainst': 2, 'Mixed': 3, -1: 3,
    'ModeratelyFor': 4, 'StronglyFor': 5, 'VeryStronglyFor': 6}} for col in voting_records.columns]
encoder = OrdinalEncoder(return_df=True, mapping=mapping)
voting_records = encoder.fit_transform(voting_records)

# Reduce to two dimensions
pca = PCA(2)
voting_records = pd.DataFrame(pca.fit_transform(voting_records), index=voting_records.index,
                              columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left

# Add party of each politician
df = pd.read_csv('dataset/au_parliament_members_data.csv', index_col='Name')
voting_records['Party'] = df.loc[[name for name in voting_records.index], 'Party']
# todo: do pca for only one club to remove influence of different clubs

# Create scatter plot for reduced dimensions
sns.scatterplot(data=voting_records, x='Component 1', y='Component 2', hue='Party')

# Create pairplot of all the variables with hue set to class
# sns.pairplot(voting_records.iloc[:, 0:5])
# plt.show()
# pd.plotting.scatter_matrix(voting_records.loc[:, :8], alpha = 0.2, figsize = (6, 6), diagonal = 'kde')

# fig, ax = plt.subplots()
# ax.plot(t, s)
#
# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()
#
# fig.savefig("test.png")
# plt.show()