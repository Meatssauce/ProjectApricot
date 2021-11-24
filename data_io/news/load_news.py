from data_io.api import DataType, Format, SortingPolicy, to_timestamp, overwrite_query
import webhoseio
overwrite_query()

webhoseio.config(token="12fede3e-49db-40a0-adc9-62d69ba00005")
query_params = {
    "keywords": "\"Mark Dreyfus\"",  # use quotation to look for exact matches
    "language": "english",
    "thread.country": "AU",
    "category": "politics",
    "sort": SortingPolicy.get_default().value,
    "format": Format.JSON.value,
    "ts": to_timestamp(days_ago=15)
}

output = webhoseio.query(DataType.NEWS_BLOGS_AND_FORMS.value, query_params)
print(output['posts'][0]['text'])  # Print the text of the first post
print(output['posts'][0]['published'])  # Print the text of the first post publication date

# Get the next batch of posts
output = webhoseio.get_next()
if output['posts']:
    print(output['posts'][0]['thread']['site'])  # Print the site of the first post
else:
    print(output)
