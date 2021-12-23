import pandas as pd
import os
from definitions import ROOT_DIR

data_dir = os.path.join(ROOT_DIR, 'datasets', 'tvfy',)
data = pd.read_json(os.path.join(data_dir, "politicians_and_policies_sample.json"))

latest_member = pd.json_normalize(data["latest_member"]).rename({'id': 'latest_member.id'}, axis=1)
general_stats = pd.concat([data.drop(columns=['latest_member', 'offices', 'policy_comparisons']), latest_member], axis=1)

offices = data[['id', 'offices']].explode('offices', ignore_index=True)
offices = pd.concat([offices['id'], pd.json_normalize(offices['offices'])], axis=1)

policy_comparisons = data[['id', 'policy_comparisons']].explode('policy_comparisons', ignore_index=True)
policy_comparisons = pd.concat([policy_comparisons['id'], pd.json_normalize(policy_comparisons['policy_comparisons'])], axis=1)

general_stats.to_csv(os.path.join(data_dir, 'general_stats.csv'), index=False)
offices.to_csv(os.path.join(data_dir, 'offices.csv'), index=False)
policy_comparisons.to_csv(os.path.join(data_dir, 'policy_comparisons.csv'), index=False)
