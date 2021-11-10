import os
from typing import Tuple, FrozenSet

import pandas as pd
import numpy as np
import nlpaug.flow as naf
import nlpaug.augmenter.word as naw
from functools import cache


def reduce_subclasses(annotated_texts: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
    """Reduce number of classes by merging homogenous(no conflicting) subcategories of main classes."""
    n_classes = len(annotated_texts['label'].unique())
    data_size = len(annotated_texts)
    # codes whose subcategories are so similar that they can be disregarded
    homogenous_categories = ['601', '602', '606', '607', '201', '416', '608', '103']

    category_prefixes = annotated_texts['label'].str.extract(r'(\d+)\..', expand=False)
    annotated_texts['label'] = np.where(
        category_prefixes.isin(homogenous_categories),
        category_prefixes,
        annotated_texts['label']
    )

    if verbose > 0:
        print(f"Merged subcategories for {homogenous_categories}\n"
              f"Number of classes: {n_classes} -> {len(annotated_texts['label'].unique())}\n"
              f"Data size: {data_size} -> {len(annotated_texts)}")

    return annotated_texts


def keep_top_k_classes(annotated_texts: pd.DataFrame, k: int, verbose: int = 0) -> pd.DataFrame:
    """Keep only top k most frequent classes, the rest are changed to 000."""
    n_classes = len(annotated_texts['label'].unique())
    top_k_classes = annotated_texts['label'].value_counts().index[:k]
    annotated_texts['label'] = np.where(annotated_texts['label'].isin(top_k_classes), annotated_texts['label'], "000")
    if verbose > 0:
        print(f"Kept top {k} classes: {top_k_classes.to_list()}. Set {n_classes} others to 000")
    return annotated_texts


def random_undersample(annotated_texts: pd.DataFrame, random_state: int = None, verbose: int = 0) -> pd.DataFrame:
    """Random under sample all majority classes"""
    distribution = annotated_texts['label'].value_counts().describe()
    if verbose > 0:
        print(f"Under-sampling to {distribution['min']} samples per class.")
    return annotated_texts.groupby('label').sample(n=int(distribution['min']), random_state=random_state)


def augment(annotated_texts: pd.DataFrame, batch_size: int = 32, max_length: int = 512,
            verbose: int = 0) -> pd.DataFrame:
    """Performs text augmentation"""
    # pipe = naf.Sequential([
    #     # naw.back_translation.BackTranslationAug(max_length=max_length, batch_size=batch_size, verbose=verbose,
    #     #                                         device='cuda'),
    #     naw.ContextualWordEmbsAug(model_path='bert-base-cased', action="insert", batch_size=batch_size,
    #                               verbose=verbose, device='cuda'),
    #     naw.split.SplitAug(aug_p=0.3, min_char=2, verbose=verbose)
    # ])

    # Truncate
    annotated_texts = annotated_texts.copy()
    annotated_texts['text'] = np.where(annotated_texts['text'].str.len() > max_length,
                                       annotated_texts['text'].str[:max_length],
                                       annotated_texts['text'])
    # Augment
    pipe = naf.Sequential([
        naf.Sometimes([
            naw.ContextualWordEmbsAug(aug_p=0.3, model_path='bert-base-cased', action="insert",
                                      batch_size=batch_size, verbose=verbose, device='cuda'),
            naw.ContextualWordEmbsAug(aug_p=0.3, model_path='bert-base-cased', action="substitute",
                                      batch_size=batch_size, verbose=verbose, device='cuda')
        ]),
        naw.split.SplitAug(aug_p=0.3, min_char=2, verbose=verbose)
    ])
    annotated_texts['text'] = pipe.augment(annotated_texts['text'].to_list())

    return annotated_texts
    # augmenters = [
    #     naw.ContextualWordEmbsAug(aug_p=0.3, model_path='bert-base-cased', action="insert",
    #                               batch_size=batch_size, verbose=verbose, device='cuda'),
    #     naw.ContextualWordEmbsAug(aug_p=0.3, model_path='bert-base-cased', action="substitution",
    #                               batch_size=batch_size, verbose=verbose, device='cuda'),
    #     naw.split.SplitAug(aug_p=0.3, min_char=2, verbose=verbose)
    # ]
    # results = []
    # for augmenter in augmenters:
    #     result = annotated_texts.copy()
    #     result['text'] = augmenter.augment(result['text'].to_list())
    #     results.append(result)
    #
    # # Merge append augmented data
    # augmented_texts = pd.concat([annotated_texts] + results, ignore_index=True)

    # if verbose >= 1:
    #     i = 0
    #     for pre, post in zip(annotated_texts['text'], results[0]['text']):
    #         print('pre:\n' + pre)
    #         print('post:\n' + post)
    #         i += 1
    #         if i >= 5:
    #             break

    # return augmented_texts


@cache
def load_data(countries: FrozenSet[str] = frozenset({'AU', 'CA', 'IE', 'IL', 'NZ', 'SA', 'UK', 'US'}),
              return_raw=False, data_dir=None) -> pd.DataFrame:
    """Load annotated text data from disk and performs basic preprocessing."""

    def read_and_tag_csv(path, country):
        df = pd.read_csv(path)
        df['country'] = country
        return df

    # Load annotated text from MARPOR corpus
    data_dir = data_dir or os.path.join('../datasets', 'MARPOR', 'Annotated text')
    country_data_dirs = {country: os.path.join(data_dir, f'{country} 2001-2021')
                         for country in countries}
    annotated_texts_data = [
        read_and_tag_csv(full_path, country)
        for country, directory in country_data_dirs.items()
        for filename in os.listdir(directory)
        if os.path.isfile(full_path := os.path.join(directory, filename))
    ]
    annotated_texts = pd.concat(annotated_texts_data, axis=0, ignore_index=True)

    if return_raw:
        # Return dataframe without basic preprocessing
        return annotated_texts

    # Basic preprocessing
    annotated_texts = (
        annotated_texts.rename(columns={'cmp_code': 'label'})
        .drop(columns=['eu_code'])
    )
    annotated_texts = annotated_texts[annotated_texts['label'] != 'H']  # drop headings
    annotated_texts['label'] = (
        annotated_texts['label'].astype(str)
        .str.replace('.0', '', regex=False)  # remove redundant suffix
        .str.replace(r'^0$', '000', regex=True)  # political statements without clear category
        .str.replace('nan', 'N/A', regex=False)  # non-political statements
    )

    return annotated_texts
