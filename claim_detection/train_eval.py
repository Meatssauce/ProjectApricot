# Attempt to buiild a claim detector from MARPOR corpus

import os
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder, OneHotEncoder
from joblib import dump
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from ultils import load_data, reduce_subclasses, keep_top_k_classes, random_undersample, augment


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


def train_eval(pretrained_model: str, annotated_texts: pd.DataFrame = None, max_length: int = 512):
    # Create folders to store results
    model_dir = os.path.join('fine-tuned-models', pretrained_model.replace('/', '-'))
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    annotated_texts = annotated_texts if annotated_texts is not None else load_data()
    annotated_texts = reduce_subclasses(annotated_texts, verbose=1)
    annotated_texts = keep_top_k_classes(annotated_texts, k=22, verbose=1)
    annotated_texts = random_undersample(annotated_texts, random_state=1, verbose=1)

    # Split dataframe into train, validation and test, 6:2:2
    y, X = annotated_texts[['label']], annotated_texts.drop(columns=['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=1, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
    print(len(X_train), 'train examples')
    print(len(X_val), 'validation examples')
    print(len(X_test), 'test examples')

    # Augment text
    # X_train = augment(X_train, batch_size=32, max_length=max_length, verbose=1)

    # Encode label
    # label_encoder = BinaryEncoder()
    label_encoder = OneHotEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_val = label_encoder.transform(y_val)
    if not isinstance(y_train, pd.Series) and y_train.shape[1] > 1:
        num_classes = y_train.shape[1]
    else:
        num_classes = len(annotated_texts['label'].unique())

    with open(os.path.join(model_dir, 'label_encoder.joblib'), 'wb') as f:
        dump(label_encoder, f)

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = TFAutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_classes)

    # Reduce max input token count to save memory at the cost of accuracy
    tokenizer.model_max_length = max_length
    # default to right padding for model with absolute position embeddings
    tokenizer.padding_side = "right"

    # Add special tokens
    # special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]'}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))
    # # fix model padding token id
    # model.config.pad_token_id = tokenizer.pad_token
    # model.config.eos_token_id = tokenizer.eos_token

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
        metrics=[tf.metrics.CategoricalAccuracy()],
    )
    # error when using f1_marco
    # NotImplementedError: Cannot convert a symbolic Tensor (tf_roberta_for_sequence_classification/classifier/out_proj/BiasAdd:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1,
                                                      restore_best_weights=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training_1/cp.ckpt", save_weights_only=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=8, epochs=5,
                        callbacks=[early_stopping, cp_callback])
    model.save(model_dir)
    # model = tf.keras.models.load_model(os.path.join('fine-tuned-models', 'from-gpu-cloud', 'distilroberta-base',
    #                                                 'tf_model.h5'))
    scores = model.evaluate(X_test, y_test)

    with open(os.path.join(model_dir, 'train-history.joblib'), 'wb') as logs_file, \
            open(os.path.join(model_dir, 'scores.joblib'), 'wb') as scores_file:
        dump(scores, scores_file)
        print(history)
        dump(history.histroy, logs_file)


def main(annotated_texts: pd.DataFrame = None):
    # cached: EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, gpt2-medium, gpt2-large, bert-base-cased
    pretrained_models = ['bert-base-cased', 'roberta-base', 'distilroberta-base']
    for model in pretrained_models:
        train_eval(model, annotated_texts=annotated_texts, max_length=256)


if __name__ == '__main__':
    main()
