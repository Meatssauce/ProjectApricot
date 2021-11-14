import os
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import pyperclip as pc
from ultils import load_data, keep_top_k_classes, reduce_subclasses, augment, random_undersample


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


def save_class_count_plot(annotated_texts: pd.DataFrame, filename: str = 'class-count-plot.png'):
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
    plt.savefig(filename)


def get_half_annotated(annotated_texts: pd.DataFrame) -> pd.DataFrame:
    """Get examples where only the parent category is annotated"""
    sub_categories = annotated_texts['label'][annotated_texts['label'].str.contains(r'\.[^0]', regex=True)]
    parent_categories = sub_categories.str[:-2].unique()
    half_annotated_texts = annotated_texts[annotated_texts['label'].isin(parent_categories)]
    return half_annotated_texts


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
    half_annotated_texts = half_annotated_texts.sort_values(by=['freq', 'label'], ascending=[False, True]) \
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
    annotated_texts = load_data()
    # save_report(augmented_texts)
    save_class_count_plot(annotated_texts)

    annotated_texts = reduce_subclasses(annotated_texts, verbose=1)
    annotated_texts = keep_top_k_classes(annotated_texts, k=20, plus=['N/A'], other='000', verbose=1)
    annotated_texts = random_undersample(annotated_texts, random_state=1, verbose=1)
    save_class_count_plot(annotated_texts, filename='class-count-plot-reduced-balanced.png')

    # augmented_texts = augment(augmented_texts, verbose=1)
    # save_class_count_plot(augmented_texts, filename='class-count-plot-reduced-balanced-augmented.png')

    # print(augmented_texts['label'].value_counts().describe())  # remove bottom 25% of classes?
    # augmented_texts = reduce_subclasses(augmented_texts)
    # print(augmented_texts['label'].value_counts().describe())
    # continue_labelling(augmented_texts)


if __name__ == '__main__':
    main()
