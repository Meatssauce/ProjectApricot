import os
from typing import Tuple, FrozenSet, List

import pandas as pd
import numpy as np
import nlpaug.flow as naf
import nlpaug.augmenter.word as naw
from functools import cache
import ujson

from tqdm import tqdm
from wordcloud import WordCloud
import zipfile as zf
import shutil
# from textaugment import Word2vec
# import gensim.downloader as downloader


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


def keep_top_k_classes(annotated_texts: pd.DataFrame, k: int, plus: List[str] = None, other: str = None,
                       verbose: int = 0) -> pd.DataFrame:
    """
    Keep only top k most frequent classes plus specified classes, the rest are changed to a specific class.

    :param annotated_texts:
    :param k: top k classes to keep
    :param plus: also keep these classes
    :param other: change the rest into this class
    :param verbose: set to 1 to show more info
    :return:
    """
    n_classes = len(annotated_texts['label'].unique())
    if plus is None:
        plus = []
    top_classes = [class_ for class_ in annotated_texts['label'].value_counts().index
                   if class_ not in plus + [other]][:k] + plus
    annotated_texts['label'] = np.where(annotated_texts['label'].isin(top_classes), annotated_texts['label'], other)

    if verbose > 0:
        additional = f'Plus {plus}.' if plus else ''
        print(f"Kept top {k} classes: {top_classes}. {additional} Set {n_classes} others to {other}")

    return annotated_texts


def random_undersample(annotated_texts: pd.DataFrame, random_state: int = None, verbose: int = 0) -> pd.DataFrame:
    """Random under sample all majority classes"""
    distribution = annotated_texts['label'].value_counts().describe()
    if verbose > 0:
        print(f"Under-sampling to {distribution['min']} samples per class.")
    return annotated_texts.groupby('label').sample(n=int(distribution['min']), random_state=random_state)


def augment(annotated_texts: pd.DataFrame, batch_size: int = 32, max_length: int = 512, device: str = 'cpu',
            verbose: int = 0, drop_original: bool = False) -> pd.DataFrame:
    """ Performs text augmentation

    :param annotated_texts: training data
    :param batch_size:
    :param max_length:
    :param device: 'cpu' or 'cuda'
    :param verbose:
    :param drop_original: set to True to return only augmented data
    :return:
    """
    # pipe = naf.Sequential([
    #     # naw.back_translation.BackTranslationAug(max_length=max_length, batch_size=batch_size, verbose=verbose,
    #     #                                         device='cuda'),
    #     naw.ContextualWordEmbsAug(model_path='bert-base-cased', action="insert", batch_size=batch_size,
    #                               verbose=verbose, device='cuda'),
    #     naw.split.SplitAug(aug_p=0.3, min_char=2, verbose=verbose)
    # ])

    # Truncate
    augmented_texts = annotated_texts.copy()
    augmented_texts['text'] = np.where(augmented_texts['text'].str.len() > max_length,
                                       augmented_texts['text'].str[:max_length],
                                       augmented_texts['text'])
    # keep texts with at least two valid tokens
    augmented_texts = augmented_texts[augmented_texts['text'].str.contains(r'[a-zA-Z0-9]{2,}')]

    # Augment
    pipe = naf.Sequential([
        naf.Sometimes([
            naw.ContextualWordEmbsAug(aug_p=0.3, model_path='distilroberta-base', action="insert",
                                      batch_size=batch_size, verbose=verbose),
            naw.ContextualWordEmbsAug(aug_p=0.3, model_path='distilroberta-base', action="substitute",
                                      batch_size=batch_size, verbose=verbose),
        ]),
        # naw.SynonymAug(aug_p=0.3, verbose=verbose),
        naw.SplitAug(aug_p=0.1, verbose=verbose)
    ])
    pipe.device = device
    augmented_texts['text'] = pipe.augment(augmented_texts['text'].to_list())

    if drop_original:
        return augmented_texts
    else:
        return pd.concat([annotated_texts, augmented_texts], axis=0)
    # augmenters = [
    #     naw.ContextualWordEmbsAug(aug_p=0.3, model_path='bert-base-cased', action="insert",
    #                               batch_size=batch_size, verbose=verbose, device='cuda'),
    #     naw.ContextualWordEmbsAug(aug_p=0.3, model_path='bert-base-cased', action="substitution",
    #                               batch_size=batch_size, verbose=verbose, device='cuda'),
    #     naw.split.SplitAug(aug_p=0.3, min_char=2, verbose=verbose)
    # ]
    # results = []
    # for augmenter in augmenters:
    #     result = augmented_texts.copy()
    #     result['text'] = augmenter.augment(result['text'].to_list())
    #     result['text'].str.replace(r"\s'\s", "'", regex=True)
    #     results.append(result)

    # # Merge append augmented data
    # augmented_texts = pd.concat([augmented_texts] + results, ignore_index=True)
    #
    # if verbose >= 1:
    #     i = 0
    #     for pre, post in zip(augmented_texts['text'], results[0]['text']):
    #         print('pre:\n' + pre)
    #         print('post:\n' + post)
    #         i += 1
    #         if i >= 5:
    #             break
    #
    # return augmented_texts

    # augmentation ideas
    # cannot use sentence level augmentations we only have quasi-sentences by themselves
    # contextual embedding substitution, insertion
    # minimal to no random shuffling - it can change the meaning of a sentence
    # decent amount of word splitting - may be a frequent occurrence in scraped text
    # speech style transformations (formal to casual to very casual)
    # insertion of filler words (um, hum, like, i think, yeah, i mean, well, look)
    # abstract summarization - maybe only for examples that are too long
    # use reserved for phrase-to-phrase and phrase-to-word and word-to-phrase replacement -- use websites that do this
    # use augmentation to address class imbalance (augment minority classes first)
    # use an augmentation pipeline


def augment_book_review_data(reviews: pd.DataFrame, batch_size: int = 32, max_length: int = 512, device: str = 'cpu',
                             drop_original: bool = False, verbose: int = 0) -> pd.DataFrame:
    # regularised_substitutes = {
    #     'book': {'item', 'policy', 'idea', 'notion', 'undertaking', 'trip', 'experience', 'system'},
    #     'read': {'support', 'change', 'review'},
    #     'character': {'person', 'candidate', 'case', 'areas'},
    #     'story': {'proposal', 'plan', 'picture', 'narrative'},
    #     'author': {'entity', 'writer', 'speaker', 'person', 'company', 'people'},
    # }
    # all_reserved_tokens: List[List[str]] = [[k] + list(v) for k, v in regularised_substitutes.items()]
    # all_reserved_tokens += [[e.capitalize() for e in tokens] for tokens in all_reserved_tokens]
    # pipe = naf.Sequential([naw.ReservedAug(all_reserved_tokens)])
    #
    # augmented_reviews = reviews.copy()
    # augmented_reviews['text'] = pipe.augment(reviews['text'].to_list())
    # model = downloader.load('word2vec-google-news-300')
    # aug_w2v = naw.WordEmbsAug(
    #     model_type='word2vec',
    #     # model_path='./GoogleNews-vectors-negative300.bin',
    #     model=model,
    #     action="substitute"
    # )

    reviews = reviews.copy()
    reviews['text'] = np.where(reviews['text'].str.len() > max_length, reviews['text'].str[:max_length],
                               reviews['text'])
    # keep texts with at least two valid tokens
    augmented_reviews = reviews[reviews['text'].str.contains(r'[a-zA-Z0-9]{2,}')]

    aug_contextual = naw.ContextualWordEmbsAug(model_path='distilbert-base-cased', action='substitute', aug_p=0.5,
                                               batch_size=batch_size, verbose=verbose, device=device)
    augmented_reviews['text'] = aug_contextual.augment(reviews['text'].to_list())
    augmented_reviews['text'] = augmented_reviews['text'].str.replace(r"\s'\s", "'", regex=True)

    # Download Google Word2vec embeddings
    # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    # t = Word2vec(model=model)
    # augmented_reviews = reviews.copy()
    # augmented_reviews['text'] = [t.augment(text) for text in augmented_reviews['text']]
    # augmented_reviews['text'] = augmented_reviews['text'].str.replace(r"\s'\s", "'", regex=True)

    if drop_original:
        return augmented_reviews
    else:
        return pd.concat([reviews, augmented_reviews], axis=0)


def make_word_cloud(texts: str, filename: str = 'word-cloud.png') -> None:
    cloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800,
                      height=400)
    cloud.generate(texts)
    cloud.to_image()
    cloud.to_file(filename)


@cache
def load_data(countries: FrozenSet[str] = frozenset({'AU', 'CA', 'IE', 'IL', 'NZ', 'SA', 'UK', 'US'}),
              return_raw=False, data_dir=os.path.join('..', 'datasets', 'MARPOR', 'Annotated text')) -> pd.DataFrame:
    """Load annotated text data from disk and performs basic preprocessing."""
    def read_and_tag_csv(path, country):
        df = pd.read_csv(path)
        df['country'] = country
        return df

    # Load annotated text from MARPOR corpus
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
    annotated_texts['text'] = annotated_texts['text'].str.encode('ascii', 'ignore').str.decode('ascii')
    annotated_texts = annotated_texts.dropna(subset=['text', 'label'], how='any')

    return annotated_texts


@cache
def load_annotated_book_reviews(file_path=os.path.join('..', 'datasets', 'non-political-texts',
                                                       'goodreads_reviews_spoiler.json'),
                                return_raw=False) -> pd.DataFrame:
    """Load goodreads spoilers book review data in appropriate format for classifier."""
    # Load data
    with open(file_path, 'r') as f:
        reviews = [ujson.loads(line.rstrip()) for line in tqdm(f)]  # loads as dict from some reason
    reviews = pd.DataFrame.from_records(reviews)

    if return_raw:
        # Return dataframe without basic preprocessing
        return reviews

    # Transform to conform to input format
    reviews = reviews.rename(columns={'review_sentences': 'text'})
    reviews = reviews[['text']].explode('text')
    reviews['text'] = reviews['text'].str[1]
    reviews['label'] = 'N/A'

    # Basic preprocessing
    reviews['text'] = reviews['text'].str.encode('ascii', 'ignore').str.decode('ascii')
    reviews = reviews.dropna(subset=['text', 'label'], how='any')

    return reviews[['text', 'label']]


def inject_book_reviews(reviews: pd.DataFrame, annotated_texts: pd.DataFrame, multiplier: float = 1.0) -> pd.DataFrame:
    """Add book review data as N/A labelled rows."""
    current_size = len(annotated_texts[annotated_texts['label'] == 'N/A'])
    injection_size = min(len(reviews), int(multiplier * current_size))
    injection_df = reviews.sample(injection_size)
    return pd.concat([annotated_texts, injection_df], axis=0)


def unzip_dir(filename):
    files = zf.ZipFile(filename, 'r')
    files.extractall(filename[:-4])
    files.close()


def zip_dir(source_dir):
    shutil.make_archive(source_dir, 'zip', source_dir)


# zip_dir(os.path.join('../datasets', 'MARPOR', 'Annotated text'))
# zip_dir(os.path.join('../datasets', 'non-political-texts'))
# unzip_dir('Annotated text.zip')
# unzip_dir('non-political-texts.zip')
