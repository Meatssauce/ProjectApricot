# Attempt to build a claim classifier and detector from MARPOR corpus
# Use this classifier to find top frequently mentioned policies in public discord as well as salient regions in
# input text data

import os
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder, OneHotEncoder
from joblib import dump, load
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from ultils import load_data, reduce_subclasses, keep_top_k_classes, random_undersample, augment, inject_book_reviews, \
    load_annotated_book_reviews


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
    y_pred = np.argmax(y_pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return {
        'f1': f1,
        'accuracy': acc,
        'precision': precision,
        'recall': recall
    }


# class CategoricalTruePositives(tf.keras.metrics.Metric):
#     def __init__(self, name="f1", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.f1_score = self.add_weight(name="f1", initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
#         tp = tf.reduce_sum(tf.cast(y_true, "int32") == tf.cast(y_pred, "int32"))
#         tn =
#         values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
#         values = tf.cast(values, "float32")
#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, "float32")
#             values = tf.multiply(values, sample_weight)
#         self.f1_score.assign_add(tf.reduce_sum(values))
#
#     def result(self):
#         return self.f1_score
#
#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.f1_score.assign(0.0)


def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def ds_to_tf_ds(dataset: Dataset, shuffle: bool = False, batch_size: int = 32,
                target_name: str = 'label', features=None) -> tf.data.Dataset:
    """Convert huggingFace Dataset into Tensorflow Dataset"""
    # Remove text column which should have already been used by the tokenizer and is now redundant
    dataset = dataset.remove_columns(['text']).with_format('tensorflow')  # can we keep text column?
    features = {x: dataset[x] for x in features}
    tf_dataset = tf.data.Dataset.from_tensor_slices((features, dataset[target_name]))
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset


# def df_to_dataset(dataframe: pd.Dataframe, shuffle=False, batch_size=32) -> tf.data.Dataset:
#     """A utility method to create a tf.data dataset from a Pandas Dataframe"""
#     dataframe = dataframe.copy()
#     labels = dataframe.pop('label')
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds


def train_eval(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model: str, num_classes: int,
               max_length: int = 512):
    # Create folders to store results
    model_dir = os.path.join('final-fine-tuned-models', pretrained_model.replace('/', '-'))
    os.makedirs(model_dir, exist_ok=True)

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = TFAutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_classes)

    # Reduce max input token count to save memory at the cost of accuracy
    tokenizer.model_max_length = max_length
    # default to right padding for model with absolute position embeddings
    tokenizer.padding_side = "right"

    # Add special tokens
    special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    # # fix model padding token id
    # model.config.pad_token_id = tokenizer.pad_token

    # Convert to huggingface Dataset
    # train_ds = Dataset.from_pandas(train_df)
    # val_ds = Dataset.from_pandas(val_df)
    # test_ds = Dataset.from_pandas(test_df)

    # Tokenize data
    # train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    # val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    # test_ds = test_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    X_train = tokenizer(X_train['text'].to_list(), padding='max_length', truncation=True, return_tensors='tf').data
    X_val = tokenizer(X_val['text'].to_list(), padding='max_length', truncation=True, return_tensors='tf').data
    X_test = tokenizer(X_test['text'].to_list(), padding='max_length', truncation=True, return_tensors='tf').data

    # Convert to Tensorflow Datasets
    # batch_size = 8
    # train_ds = ds_to_tf_ds(train_ds, shuffle=True, batch_size=batch_size, features=tokenizer.model_input_names)
    # val_ds = ds_to_tf_ds(val_ds, batch_size=batch_size, features=tokenizer.model_input_names)
    # test_ds = ds_to_tf_ds(test_ds, batch_size=batch_size, features=tokenizer.model_input_names)

    # Use mixed precision to save memory
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Train classifier, evaluate and save results
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['categorical_crossentropy', tfa.metrics.F1Score(num_classes=num_classes)],
    )
    # error when using f1_marco
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1,
                                                      restore_best_weights=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training_1/cp.ckpt", save_weights_only=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=8, epochs=15,
                        callbacks=[early_stopping, cp_callback])
    model.save(model_dir)
    # model = tf.keras.models.load_model(os.path.join('fine-tuned-models', 'from-gpu-cloud', 'distilroberta-base',
    #                                                 'tf_model.h5'))
    scores = model.evaluate(X_test, y_test)

    with open(os.path.join(model_dir, 'train-history.joblib'), 'wb') as logs_file, \
            open(os.path.join(model_dir, 'scores.joblib'), 'wb') as scores_file:
        dump(scores, scores_file)
        try:
            print(history)
            dump(history, logs_file)
        except:
            pass


def main():
    max_length = 256  # set max_length to 512 if gpu has more memory else set to 256

    try:
        # Load cached data
        X_train = pd.read_csv(os.path.join('cache', 'X_train.csv'), index_col=None).fillna('')
        y_train = pd.read_csv(os.path.join('cache', 'y_train.csv'), index_col=None).fillna('')
        X_val = pd.read_csv(os.path.join('cache', 'X_val.csv'), index_col=None).fillna('')
        y_val = pd.read_csv(os.path.join('cache', 'y_val.csv'), index_col=None).fillna('')
        X_test = pd.read_csv(os.path.join('cache', 'X_test.csv'), index_col=None).fillna('')
        y_test = pd.read_csv(os.path.join('cache', 'y_test.csv'), index_col=None).fillna('')

        num_classes = len(y_train['label'].unique())

    except (FileNotFoundError, EOFError):
        # Load data
        annotated_texts = load_data()
        print("annotated_texts")
        annotated_texts.head()
        reviews = load_annotated_book_reviews()
        print("reviews")
        reviews.head()

        annotated_texts = annotated_texts.dropna(how='any')
        annotated_texts = inject_book_reviews(reviews, annotated_texts)
        annotated_texts = reduce_subclasses(annotated_texts, verbose=1)
        annotated_texts = keep_top_k_classes(annotated_texts, k=20, plus=['N/A'], other='000', verbose=1)
        annotated_texts = random_undersample(annotated_texts, random_state=1, verbose=1)

        num_classes = len(annotated_texts['label'].unique())

        # Split dataframe into train, validation and test, 6:2:2
        y, X = annotated_texts[['label']], annotated_texts.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=1, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
        print(len(X_train), 'train examples')
        print(len(X_val), 'validation examples')
        print(len(X_test), 'test examples')

        # Augment text
        # train_df = pd.concat([X_train, y_train], axis=1)
        # train_df = augment(train_df, batch_size=8, max_length=max_length, verbose=1)
        # y_train, X_train = train_df[['label']], train_df.drop(columns=['label'])
        # assert len(X_train) == len(y_train)

        # Cache preprocessed data
        cache_dir = os.path.join('cache')
        os.makedirs(cache_dir, exist_ok=True)
        X_train.to_csv(os.path.join(cache_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(cache_dir, 'y_train.csv'), index=False)
        X_val.to_csv(os.path.join(cache_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(cache_dir, 'y_val.csv'), index=False)
        X_test.to_csv(os.path.join(cache_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(cache_dir, 'y_test.csv'), index=False)

    # Encode label
    # label_encoder = BinaryEncoder()
    label_encoder = OneHotEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_val = label_encoder.transform(y_val)

    # cached: EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, gpt2-medium, gpt2-large, bert-base-cased
    # 'distilroberta-base', 'roberta-base', 'xlnet-base-cased', 'albert-xlarge-v2',
    pretrained_models = ['albert-xlarge-v2', 'xlnet-base-cased', ]
    for pretrained_model in pretrained_models:
        train_eval(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, num_classes=num_classes,
                   max_length=max_length)


if __name__ == '__main__':
    main()
