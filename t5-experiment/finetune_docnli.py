import textwrap
import torch
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_metric,
    load_from_disk,
    concatenate_datasets
)
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
import pandas as pd
from definitions import ROOT_DIR, MODEL_DIR, CHECKPOINT_DIR, TRAIN_LOGS_DIR, CACHED_DATASET_DIR, SEED, DOCNLI_DIR, \
    DOCNLI_PROCESSED_PATH
import os
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import ujson


def tokenize_dataset(dataset, tokenizer) -> DatasetDict:
    print('Begin tokenizing targets')
    with tokenizer.as_target_tokenizer():
        dataset = dataset.map(
            lambda x: tokenizer(
                x['labels'],
                padding='max_length',
                max_length=6,
                truncation=True,
                return_attention_mask=False,
                return_tensors='np'
            ),
            batched=True
        )
    dataset = dataset.rename_column('labels', 'raw_labels').rename_column('input_ids', 'labels')

    def replace_pad_token(x):
        x['labels'] = np.where(x['labels'] == tokenizer.pad_token_id, -100, x['labels'])
        return x
    dataset = dataset.map(replace_pad_token, batched=True)

    print('Begin tokenizing inputs')
    dataset = dataset.map(
        lambda x: tokenizer(
            [f'snli hypothesis: {h} premise: {p}' for h, p in zip(x['hypothesis'], x['premise'])],
            padding='max_length',
            truncation=True,
            return_tensors='np'
        ),
        batched=True,
        batch_size=128,
        num_proc=3,
    )

    return dataset


def prepare_data(tokenizer) -> DatasetDict:
    """Data preparation pipeline for DocNLI"""
    df_train = pd.read_json(os.path.join(DOCNLI_DIR, 'train.json'))
    df_dev = pd.read_json(os.path.join(DOCNLI_DIR, 'dev.json'))
    df_test = pd.read_json(os.path.join(DOCNLI_DIR, 'test.json'))

    dataset = DatasetDict(dict(
        train=Dataset.from_pandas(df_train),
        validation=Dataset.from_pandas(df_dev),
        test=Dataset.from_pandas(df_test)
    ))

    dataset = concatenate_datasets([dataset[i] for i in ['train', 'validation', 'test']])

    # Under-sampling not needed as there are large number of samples for each class

    old = set()

    def is_new(x):
        if x['premise'] + x['hypothesis'] in old:
            return False
        old.add(x['premise'] + x['hypothesis'])
        return True

    # Clean dataset
    dataset = dataset.rename_column('label', 'labels')
    dataset = dataset.filter(lambda x: is_new(x) and x['premise'] and x['hypothesis'] and x['labels'])

    # Split dataset
    train_testvalid = dataset.train_test_split(test_size=2000, shuffle=True, seed=SEED)
    test_valid = train_testvalid['test'].train_test_split(test_size=1000, shuffle=True, seed=SEED)
    dataset = DatasetDict(dict(
        train=train_testvalid['train'], validation=test_valid['train'], test=test_valid['test']))

    print(dataset)

    # Tokenize processed targets
    dataset = tokenize_dataset(dataset, tokenizer)

    return dataset


def plot_confusion_matrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    plot = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    plot.set_xlabel("Predicted label")
    plot.set_ylabel("True label")
    return plot.get_figure()


def main():
    model_name = 't5-base'
    run_name = f'{model_name} Fine-tuning on DocNLI'
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    if do_load_from_disk := True:
        tokenized_dataset = load_from_disk(DOCNLI_PROCESSED_PATH)
    else:
        tokenized_dataset = prepare_data(tokenizer)
        tokenized_dataset.save_to_disk(DOCNLI_PROCESSED_PATH)

    # Reformat dataset for pytorch
    tokenized_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset = tokenized_dataset['train'].shuffle(SEED)
    val_dataset = tokenized_dataset['validation'].shuffle(SEED)
    # test_dataset = tokenized_dataset['test'].shuffle(SEED)
    # small_train_dataset = train_dataset.select(range(1000)).shuffle(SEED)
    small_val_dataset = val_dataset.select(range(100))

    # Load model and train
    checkpoint = None
    # checkpoint = os.path.join(CHECKPOINT_DIR, run_name, 'checkpoint-1500')

    model = T5ForConditionalGeneration.from_pretrained(checkpoint if checkpoint else model_name)
    model.to(device)

    if do_train := True:
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(CHECKPOINT_DIR, run_name),
            overwrite_output_dir=True,
            logging_dir=os.path.join(TRAIN_LOGS_DIR, run_name),
            run_name=run_name,
            evaluation_strategy='steps',
            # eval_steps=1000,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            # fp16=True,
            save_total_limit=10,
            generation_max_length=5,
            generation_num_beams=2,
            remove_unused_columns=True,
            seed=SEED
        )

        # optimizer = Adafactor(
        #     model.parameters(),
        #     scale_parameter=True,
        #     relative_step=True,
        #     warmup_init=True,
        #     lr=None
        # )
        # lr_scheduler = AdafactorSchedule(optimizer)
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # optimizers=(optimizer, lr_scheduler),
            optimizers=(optimizer, None)
        )
        trainer.train(resume_from_checkpoint=bool(checkpoint))

    # # Generate results
    # y_pred = model.generate(
    #     input_ids=small_val_dataset['input_ids'].to(device),
    #     attention_mask=small_val_dataset['attention_mask'].to(device)
    # )
    # y_pred = [tokenizer.decode(i, skip_special_tokens=True) for i in y_pred]
    #
    # # Display input, prediction and labels
    # inputs = [tokenizer.decode(i, skip_special_tokens=True) for i in small_val_dataset['input_ids']]
    # y_true = torch.tensor(small_val_dataset['labels'])
    # y_true[y_true == -100] = tokenizer.pad_token_id
    # y_true = [tokenizer.decode(i, skip_special_tokens=True) for i in y_true]
    #
    # for input_, label, prediction in zip(inputs, y_true, y_pred):
    #     lines = textwrap.wrap(f'Inputs:\n{input_}\n', width=100)
    #     print("\n".join(lines))
    #     print(f'\nActual label: {label}')
    #     print(f'Predicted label: {prediction}')
    #     print("=====================================================================\n")

    # Compute metrics
    y_pred = model.generate(
        input_ids=small_val_dataset['input_ids'].to(device),
        attention_mask=small_val_dataset['attention_mask'].to(device)
    )
    y_pred = [tokenizer.decode(i, skip_special_tokens=True) for i in y_pred]
    y_pred = [small_val_dataset['raw_labels'].index(i) for i in y_pred]
    y_true = [small_val_dataset['raw_labels'].index(i) for i in small_val_dataset['raw_labels']]

    score = f1_score(y_true, y_pred, average='macro')
    print(f'f1 score: {score}')

    fig = plot_confusion_matrix(y_true, y_pred)
    fig.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))


if __name__ == '__main__':
    main()

# todo: train and make sure does not terminate prematurely due to cuda error
