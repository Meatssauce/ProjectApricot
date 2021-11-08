import os
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from functools import cache
import pyperclip as pc


def save_report(annotated_texts: pd.DataFrame):
    profile = ProfileReport(
        annotated_texts,
        variables={
            'descriptions': {
                'text': 'One quasi sentence or title',
                'label': 'Code for the corresponding political topic'
            }
        }
    )
    profile.to_file('Profile Report - MARPOR AU 2001-2021.html')


def augment(annotated_texts: pd.DataFrame) -> pd.DataFrame:
    augmented_texts = annotated_texts.copy()
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-cased', action="insert", batch_size=32)

    augmented_texts['text'] = aug.augment(augmented_texts['text'].to_list())
    augmented_texts = augmented_texts[augmented_texts != annotated_texts]
    annotated_texts = pd.concat([annotated_texts, augmented_texts])

    i = 0
    for pre, post in zip(annotated_texts['text'], annotated_texts):
        print('pre:\n' + pre)
        print('post:\n' + post)
        i += 1
        if i >= 5:
            break

    return annotated_texts


@cache
def load_data(countries: Tuple[str], raw=False) -> pd.DataFrame:
    def read_and_tag_csv(path, country):
        df = pd.read_csv(path)
        df['country'] = country
        return df

    # Load annotated text from MARPOR corpus
    country_data_dirs = {country: os.path.join('../datasets', 'MARPOR', 'Annotated text', f'{country} 2001-2021')
                         for country in countries}
    annotated_texts_data = [
        read_and_tag_csv(full_path, country)
        for country, directory in country_data_dirs.items()
        for filename in os.listdir(directory)
        if os.path.isfile(full_path := os.path.join(directory, filename))
    ]
    annotated_texts = pd.concat(annotated_texts_data, axis=0, ignore_index=True)
    if raw:
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


def save_class_count_plot(annotated_texts: pd.DataFrame):
    plt.figure(figsize=(16, 6))
    ax = sns.countplot(data=annotated_texts, x='label', order=annotated_texts['label'].value_counts().index)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        # fontweight='light',
        fontsize='10'
    )
    # plt.tight_layout()
    plt.savefig('class-count-plot.png')


def get_half_annotated(annotated_texts: pd.DataFrame) -> pd.DataFrame:
    """Get examples where only the parent category is annotated"""
    sub_categories = annotated_texts['label'][annotated_texts['label'].str.contains(r'\.[^0]', regex=True)]
    parent_categories = sub_categories.str[:-2].unique()
    half_annotated_texts = annotated_texts[annotated_texts['label'].isin(parent_categories)]
    return half_annotated_texts

# similar codes without conflicting subcategories [601 602 606 607 201 416 608 103]
# can disregard subcategories for them


def reduce_classes(annotated_texts: pd.DataFrame) -> pd.DataFrame:
    """Reduce number of classes by merging homogenous(no conflicting) subcategories of main classes."""
    n_classes = len(annotated_texts['label'].unique())
    data_size = len(annotated_texts)
    homogenous_categories = ['601', '602', '606', '607', '201', '416', '608', '103']

    category_prefixes = annotated_texts['label'].str.extract(r'(\d+)\..', expand=False)
    annotated_texts['label'] = np.where(
        category_prefixes.isin(homogenous_categories),
        category_prefixes,
        annotated_texts['label']
    )

    print(f'merged subcategories for {homogenous_categories}')
    print(f"number of classes: {n_classes} -> {len(annotated_texts['label'].unique())}, "
          f"data size: {data_size} -> {len(annotated_texts)}")

    return annotated_texts


def fit_str_to_width(string: str, width: int = 80) -> str:
    new_input = ""
    for i, letter in enumerate(string):
        if i % width == 0 and i != 0:
            new_input += '\n'
        new_input += letter
    return new_input


def continue_labelling(annotated_texts: pd.DataFrame, start_at: int = 0):
    """Manually label half-annotated categories"""
    half_annotated_texts = get_half_annotated(annotated_texts)
    half_annotated_texts = half_annotated_texts.assign(freq=half_annotated_texts.groupby('label')['label']
                                                       .transform('count'))
    half_annotated_texts = half_annotated_texts.sort_values(by=['freq', 'label'], ascending=[False, True])\
        .reset_index(drop=True)
    subcategories = []

    try:
        for i, row in half_annotated_texts[start_at:].iterrows():
            print(f"ENTRY {int(i) + 1} of {len(half_annotated_texts)}")
            print(fit_str_to_width(row['text']))
            print("---------------------------------")
            print(f"current label: {row['label']}")
            pc.copy(int(row['label']))
            while True:
                try:
                    subcategory = int(input("\n\nEnter subcategory (integer only, enter -1 to quit): "))
                except ValueError as e:
                    print(f"Error. Please enter an integer.")
                else:
                    if subcategory == -1:
                        raise KeyboardInterrupt
                    break
            subcategories.append(subcategory)
            print("=================================")
    except KeyboardInterrupt:
        pass
    finally:
        print('existing...')
        padding = (len(half_annotated_texts) - len(subcategories)) * [np.nan]
        half_annotated_texts['responses'] = subcategories + padding
        half_annotated_texts.to_csv('fully-annotated-texts.csv')


def main():
    annotated_texts = load_data(('AU', 'CA', 'IE', 'IL', 'NZ', 'SA', 'UK', 'US'))
    annotated_texts.to_csv('annotated_texts.csv', index=False)

    print(annotated_texts['label'].value_counts().describe())  # remove bottom 25% of classes?
    annotated_texts = reduce_classes(annotated_texts)
    print(annotated_texts['label'].value_counts().describe())
    annotated_texts.to_csv('annotated_texts_reduced.csv', index=False)
    # continue_labelling(annotated_texts)

    # save_report(annotated_texts)
    save_class_count_plot(annotated_texts)
    # augment(annotated_texts)


if __name__ == '__main__':
    main()
