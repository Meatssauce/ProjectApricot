import json
from data_io.policies.tvfy import TvfyData, PRIVATE_API_KEY
from definitions import ROOT_DIR
import os


# Downloads all data about all politicians from TheyVoteForYou.org and save them in datasets
# The data contains policy comparisons for each politician

client = TvfyData(PRIVATE_API_KEY)
politicians_data = client.get_all_personal_details()

output_dir = os.path.join(ROOT_DIR, 'datasets', 'tvfy')

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'politicians_and_policies.json'), 'w') as f:
    json.dump(politicians_data, f)
