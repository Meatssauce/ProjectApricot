# Attempt to build a claim classifier and detector from MARPOR corpus
# Use this classifier to find top frequently mentioned policies in public discord as well as salient regions in
# input text data

import os
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder, OneHotEncoder
from joblib import dump, load
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, load_metric
from ultils import load_data, reduce_subclasses, keep_top_k_classes, random_undersample, augment, \
    inject_book_reviews, load_annotated_book_reviews


metric = load_metric("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, return_tensors='pt')


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


def df_to_dataset(dataframe: pd.DataFrame) -> tf.data.Dataset:
    """A utility method to create a tf.data dataset from a Pandas Dataframe"""
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    return ds


def train_eval(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model: str, num_classes: int,
               max_length: int = 512):
    # Create folders to store results
    model_dir = os.path.join('fine-tuned-models', pretrained_model.replace('/', '-'))
    os.makedirs(model_dir, exist_ok=True)

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_classes)

    # Reduce max input token count to save memory at the cost of accuracy
    tokenizer.model_max_length = max_length
    # default to right padding for model with absolute position embeddings
    tokenizer.padding_side = "right"

    # Add special tokens
    special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Add special tokens
    # tokenizer.pad_token = tokenizer.eos_token
    # model.resize_token_embeddings(len(tokenizer))
    # # fix model padding token id
    # model.config.pad_token_id = tokenizer.pad_token

    # Convert to huggingface Dataset
    train_ds = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
    val_ds = Dataset.from_pandas(pd.concat([X_val, y_val], axis=1))
    test_ds = Dataset.from_pandas(pd.concat([X_test, y_test], axis=1))

    # Tokenize data
    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True).shuffle(42)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True).shuffle(42)
    test_ds = test_ds.map(lambda x: tokenize(x, tokenizer), batched=True).shuffle(42)

    # Convert to Tensorflow Datasets
    # batch_size = 8
    # train_ds = ds_to_tf_ds(train_ds, shuffle=True, batch_size=batch_size, features=tokenizer.model_input_names)
    # val_ds = ds_to_tf_ds(val_ds, batch_size=batch_size, features=tokenizer.model_input_names)
    # test_ds = ds_to_tf_ds(test_ds, batch_size=batch_size, features=tokenizer.model_input_names)

    training_args = TrainingArguments(
        model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="step",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    # trainer.save_metrics()
    scores = trainer.evaluate(test_ds)

    with open(os.path.join(model_dir, 'train-history.joblib'), 'wb') as logs_file, \
            open(os.path.join(model_dir, 'scores.joblib'), 'wb') as scores_file:
        dump(scores, scores_file)


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
    # label_encoder = OneHotEncoder()
    # y_train = label_encoder.fit_transform(y_train)
    # y_test = label_encoder.transform(y_test)
    # y_val = label_encoder.transform(y_val)

    # cached: EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, gpt2-medium, gpt2-large, bert-base-cased
    # 'distilroberta-base', 'roberta-base', 'xlnet-base-cased', 'albert-xlarge-v2',
    pretrained_models = ['EleutherAI/gpt-neo-1.3B']
    for pretrained_model in pretrained_models:
        train_eval(X_train, y_train, X_val, y_val, X_test, y_test, pretrained_model, num_classes=num_classes,
                   max_length=max_length)


if __name__ == '__main__':
    main()
