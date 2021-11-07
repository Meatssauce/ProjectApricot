import os
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from functools import cache


def save_report(annotated_texts):
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


def augment(annotated_texts):
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
def load_data(countries: Tuple[str]):
    # Load annotated text from MARPOR corpus
    data_dirs = [os.path.join('../datasets', 'MARPOR', 'Annotated text', f'{country} 2001-2021')
                 for country in countries]
    annotated_texts_data = [
        pd.read_csv(full_path)
        for directory in data_dirs
        for filename in os.listdir(directory)
        if os.path.isfile(full_path := os.path.join(directory, filename))
    ]

    # Preprocessing
    annotated_texts = (
        pd.concat(annotated_texts_data, axis=0, ignore_index=True)
        .rename(columns={'cmp_code': 'label'})
        .drop(columns=['eu_code'])
    )
    annotated_texts = annotated_texts[annotated_texts['label'] != 'H']  # drop headings
    annotated_texts['label'] = (
        annotated_texts['label'].astype('float32')
        .mul(10)  # remove decimals while maintaining ordinal relationship of category codes
        .fillna(-1)  # replace nan with distinct category
        .astype('object')
    )
    annotated_texts['label'] = ['c' + str(i)[:-2] for i in annotated_texts['label'] if not np.isnan(i)]

    return annotated_texts


def save_class_count_plot(annotated_texts):
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=annotated_texts, x='label', order=annotated_texts['label'].value_counts().index)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        # fontweight='light',
        fontsize='7'
    )
    # plt.tight_layout()
    plt.savefig('class-count-plot.png')


def main():
    annotated_texts = load_data(('AU', 'CA', 'IE', 'IL', 'NZ', 'SA', 'UK', 'US'))
    # save_report(annotated_texts)
    save_class_count_plot(annotated_texts)
    # augment(annotated_texts)


if __name__ == '__main__':
    main()
