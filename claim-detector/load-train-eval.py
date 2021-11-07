## Attempt to buiild a claim detector from MARPOR corpus

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from joblib import dump
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'f1': f1,
        'accuracy': acc,
        'precision': precision,
        'recall': recall
    }


def f1_macro(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return {
        'f1': f1,
        'accuracy': acc,
        'precision': precision,
        'recall': recall
    }


def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def ds_to_tf_ds(dataset: Dataset, shuffle: bool = False, batch_size: int = 32,
                target_name: str = 'label') -> tf.data.Dataset:
    """Convert huggingFace Dataset into Tensorflow Dataset"""
    # Remove text column which should have already been used by the tokenizer and is now redundant
    dataset = dataset.remove_columns(['text']).with_format('tensorflow')  # can we keep text column?
    features = {x: dataset[x] for x in tokenizer.model_input_names}
    tf_dataset = tf.data.Dataset.from_tensor_slices((features, dataset[target_name]))
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset


# Parameters
country = 'AU'
# cached: EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, gpt2-medium, gpt2-large, bert-base-cased
pretrained_model = "bert-base-cased"

# Create folders to store results
model_dir = os.path.join('fine-tuned-models', pretrained_model.replace('/', '-'))
logs_dir = os.path.join(model_dir, 'logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Load annotated text data from MARPOR corpus
data_dir = os.path.join('../datasets', 'MARPOR', 'Annotated text', f'{country} 2001-2021')
annotated_texts_data = [pd.read_csv(full_path) for path in os.listdir(data_dir)
                        if os.path.isfile(full_path := os.path.join(data_dir, path))]

# Preprocessing
annotated_texts = (
    pd.concat(annotated_texts_data, axis=0)
    .reset_index(drop=True)
    .rename(columns={'cmp_code': 'label'})
    .drop(columns=['eu_code'])
)
annotated_texts = annotated_texts[annotated_texts['label'] != 'H']  # drop headings
annotated_texts['label'] = (
    annotated_texts['label'].astype('float32')
    .mul(10)  # remove decimals while maintaining ordinal relationship of category codes
    .fillna(-1)  # replace nan with distinct category
    .astype('int32')
)

# Augmentation
import nlpaug
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")
augmented_text = aug.augment(text)
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

label_encoder = OrdinalEncoder()
annotated_texts['label'] = label_encoder.fit_transform(annotated_texts[['label']])
num_classes = len(annotated_texts['label'].unique())

with open('label_encoder.joblib', 'wb') as f:
    dump(label_encoder, f)
# todo: save label_encoder or integrate it into a pipeline

# Split dataframe into train, validation and test, 6:2:2
train_df, test_df = train_test_split(annotated_texts, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.25)
print(len(train_df), 'train examples')
print(len(val_df), 'validation examples')
print(len(test_df), 'test examples')

# def df_to_dataset(dataframe: pd.Dataframe, shuffle=False, batch_size=32) -> tf.data.Dataset:
#     """A utility method to create a tf.data dataset from a Pandas Dataframe"""
#     dataframe = dataframe.copy()
#     labels = dataframe.pop('label')
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds
#
#
# # Convert to tensorflow datasets
# batch_size = 32  # A small batch sized 5 is used for demonstration purposes
# train_ds = df_to_dataset(train_df, batch_size=batch_size)
# val_ds = df_to_dataset(val_df, batch_size=batch_size)
# test_ds = df_to_dataset(test_df, batch_size=batch_size)


# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('A batch of texts:', feature_batch['text'])
#     print('A batch of label:', label_batch)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

# Tokenize inputs
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.model_max_length = 256
model = TFAutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_classes)

special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Convert to Tensorflow Datasets
batch_size = 8
train_ds = ds_to_tf_ds(train_ds, shuffle=True, batch_size=batch_size)
val_ds = ds_to_tf_ds(val_ds, batch_size=batch_size)
test_ds = ds_to_tf_ds(test_ds, batch_size=batch_size)

tf.keras.backend.set_floatx('float16')

# Train classifier, evaluate and save results
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()],
)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=1, verbose=1,
    mode='auto', restore_best_weights=True
)
history = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[early_stopping])
model.save_pretrained(model_dir)
scores = model.evaluate(test_ds)

with open(os.path.join(logs_dir, 'train-history.joblib'), 'wb') as logs_file, \
        open(os.path.join(logs_dir, 'scores.joblib'), 'wb') as scores_file:
    dump(history, logs_file)
    dump(scores, scores_file)
