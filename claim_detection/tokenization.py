from typing import List

import pandas as pd
import numpy as np
from joblib import load, dump


from nltk.tokenize import sent_tokenize
text = "God is Great! I won a lottery."
print(sent_tokenize(text))

df = pd.DataFrame({'article_id': [4513, 4516], 'text': ['some content.']})
df['text'] = df['text'].apply(sent_tokenize).reset_index(drop=True)
df = df.explode('text').reset_index(drop=True)

X = df[['text']]
y = ....predict(df[['text']]).argmax(...)
df['label'] = y


def mask_around(index: List[int], radius: int) -> List[int]:
    """Expand an index list to include k neighbouring index of each element of the list

    :param index: list of indexes taken from an index in the form of list(range(n))
    :param radius: number of neighbouring indexes to keep on each side
    :return:
    """
    pass
