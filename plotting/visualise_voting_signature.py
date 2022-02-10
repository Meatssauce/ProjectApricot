import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from definitions import ROOT_DIR

data = pd.read_json(os.path.join(ROOT_DIR, 'datasets/tvfy/politicians_and_policies.json'))

voting_records = data[['id', 'policy_comparisons']].explode('policy_comparisons', ignore_index=True)
voting_records = pd.concat([voting_records['id'], pd.json_normalize(voting_records['policy_comparisons'])], axis=1)

no_vote_agreement = 50
voting_records['agreement'] = np.where(voting_records['voted'], voting_records['agreement'], no_vote_agreement)
voting_records = voting_records.pivot(index='id', columns=['policy.id'], values=['agreement'])
voting_records = voting_records.fillna(no_vote_agreement)

# Reduce to 2D
pca = PCA(2)
voting_records_2d = pd.DataFrame(pca.fit_transform(voting_records), index=voting_records.index,
                                 columns=['Component 1', 'Component 2'])  # liberal vs labor and right vs left?
voting_records_2d *= -1  # for cosmetic purposes

voting_records_2d.to_csv(os.path.join(ROOT_DIR, 'datasets', 'vote-based-embedding.csv'))

sns.scatterplot(data=voting_records_2d, x='Component 1', y='Component 2')
plt.savefig(f'temp.png')
plt.close()
