import contextualSpellCheck
import pandas as pd
import numpy as np
import os
import re
from joblib import dump
from tqdm import tqdm
import json
from wordcloud import WordCloud
import pyLDAvis
from pyLDAvis import gensim_models
import gensim
import spacy
from spacy.tokens import Token
from spacy.tokenizer import _get_regex_pattern
import logging


def make_word_cloud(texts):
    cloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800,
                      height=400)
    cloud.generate(texts)
    cloud.to_image()
    if save_word_cloud:
        os.makedirs('results', exist_ok=True)
        cloud.to_file(f'results/word-cloud-{num_topics}.png')


# Parameters
# logging.basicConfig(level=logging.INFO)
min_tokens = 2
num_topics = 150
save_word_cloud = True
visualise_topics = True
# unwanted_words = {'today', 'tonight', 'now', 'will',
#                   'thank', 'congratulation', 'know', 'issue', 'new',
#                   'good', 'great', 'important',
#                   'ask', 'question', 'tell', 'talk', 'speak', 'discuss', 'start', 'think',
#                   'time', 'year', 'month',
#                   'day', 'need', 'let', 'try', 'look',
#                   've', 'll'}
unwanted_words = {'today', 'tonight', 'now', 'will', 'congratulation', 'issue',
                  'time', 'year', 'month', 'day', 'need', 'let', 'try', 'look',
                  've', 'll'}

if __name__ == '__main__':
    # Add lone alpha and html character reference detection as token attributes
    Token.set_extension('is_lone_alpha', getter=lambda token: token.is_alpha and len(token.text) == 1)
    Token.set_extension('like_html_ref', getter=lambda token: 0 < token.i < len(token.doc) - 1
                        and token.nbor(-1).text == '&' and token.nbor(1).text == ';'
                        and token.pos_ != 'PART' and not token.is_punct)

    # Load model with TextCategorizer disabled
    nlp = spacy.load('en_core_web_sm', disable=['textcat'])

    # Add contextual spelling check component to model pipeline
    # contextualSpellCheck.add_to_pipe(nlp)  # todo: find better spell check - good at short hand and insert space

    # Overwrite default tokenizer match pattern to include hashtags and @mentions
    re_token_match = fr"({_get_regex_pattern(nlp.Defaults.token_match)}|(#\w+)|(@\w+))"
    nlp.tokenizer.token_match = re.compile(re_token_match).match

    # Load all tweets as dataframe
    politicians = pd.read_csv('datasets/parliament-members.csv', index_col='Name')
    tweets_data = []
    for handle in tqdm(politicians['Twitter Handle'], desc='Loading data'):
        if handle is not np.nan:
            with open(f'datasets/tweets/{handle}.json', 'r') as f:
                for line in f:
                    tweets_data.append(json.loads(line))
    tweets = pd.DataFrame.from_records(tweets_data)
    del politicians

    print('Preprocessing')

    # Keep only unique english tweets and relevant columns
    tweets = tweets[tweets['lang'] == 'en']
    tweets = tweets.drop_duplicates(subset=['content'])
    tweets = tweets[['id', 'user', 'content']]

    # # Convert to lowercase
    # tweets.content = tweets.content.str.lower()
    #
    # # Remove urls, html tags, hashtags, @users, punctuation, extra white space and lone letters
    # tweets.content = tweets.content.str.replace(r'(https?://)?(www\.)?\w+(\.\w+)+(/\S*)?', '',
    #                                                                     regex=True)
    # tweets.content = tweets.content.str.replace(r'(https?://\S+)|(www\.\S+)|(&\S+;)', ' ',
    #                                             regex=True)
    # tweets.content = tweets.content.str.replace(r'(#[A-Za-z0-9_]+)+|(@[A-Za-z0-9_]+)+', ' ', regex=True)
    # tweets.content = tweets.content.str.replace(r'[^A-Za-z0-9]+', ' ', regex=True)
    # tweets.content = tweets.content.str.replace(r'\s{2,}|(\s+\w\s+)+', ' ', regex=True)

    # Replace morrison with scott morrison
    tweets.content = tweets.content.str.replace(r'\bscottscott\b', 'scott morrison', regex=True)
    tweets.content = tweets.content.str.replace(r'\bgovt\b', 'government', regex=True)
    tweets.content = tweets.content.str.replace(r'\bpm\b', 'prime minister', regex=True)

    # Preprocess with spacy pipeline (tokenization, stopwords removal etc)
    tokenized_tweets = []
    # for tweet in nlp.pipe(tqdm(tweets.content.to_list(), desc='Preprocessing')):
    for tweet in nlp.pipe(tweets.content.to_list()):
        tokens = [token.lemma_.lower() for token in tweet
                  if token.is_ascii
                  and not token._.is_lone_alpha
                  and not token.is_stop
                  and not token.is_punct
                  and not token.is_space
                  and not (token.like_email or token.like_url or token._.like_html_ref)
                  and not re.match(r'(#[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)', token.text)  # not a #hashtag or @mention
                  and not token.ent_type_  # not a named entity
                  and token.pos_ in ['NOUN', 'PROPN']  # include 'X'?
                  and token.lemma_ not in unwanted_words
                  ]

        # Keep only tweets with at least two tokens
        if len(tokens) >= min_tokens:
            tokenized_tweets.append(tokens)

    # Create word cloud - for data exploration purposes only
    make_word_cloud(','.join([','.join(tweet) for tweet in tokenized_tweets]))

    if visualise_topics:
        # Build LDA model
        print('Building LDA model')
        id2word = gensim.corpora.Dictionary(tokenized_tweets)
        corpus = [id2word.doc2bow(tweet) for tweet in tokenized_tweets]
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

        # Visualize the topics
        print('Visualising topics')
        os.makedirs('results', exist_ok=True)
        LDAvis_prepared = gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, f'results/ldavis-prepared-{str(num_topics)}.html')
