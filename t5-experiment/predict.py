import textwrap
from datasets import (
    load_from_disk
)
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast
)
import numpy as np
from definitions import ROOT_DIR, MODEL_DIR, CHECKPOINT_DIR, TRAIN_LOGS_DIR, DATASET_DIR, SEED
import os
from sklearn.metrics import confusion_matrix, f1_score
from finetune import prepare_data, plot_confusion_matrix
from scipy import stats


def main():
    model_name = 't5-base'

    dataset_path = os.path.join(DATASET_DIR, 'snli_processed.dat')
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    tokenized_dataset = load_from_disk(dataset_path)['validation']

    # Reformat dataset for pytorch
    tokenized_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])

    val_dataset = tokenized_dataset.shuffle(SEED)
    small_val_dataset = val_dataset.select(range(100))

    # Load models
    checkpoints = [os.path.join(CHECKPOINT_DIR, 'current_best', i)
                   for i in ['checkpoint-14000',
                             # 'checkpoint-17500',
                             # 'checkpoint-18500'
                             ]]

    # Compute metrics
    y_preds = []
    for i in tqdm(checkpoints):
        model = T5ForConditionalGeneration.from_pretrained(i)

        y_pred = model.generate(
            input_ids=small_val_dataset['input_ids'],
            attention_mask=small_val_dataset['attention_mask']
        )
        y_pred = [tokenizer.decode(i, skip_special_tokens=True) for i in y_pred]
        y_pred = [small_val_dataset.features['raw_labels'].str2int(i) for i in y_pred]
        y_preds.append(y_pred)
    y_pred = stats.mode(np.stack(y_preds))[0][0].tolist()
    y_true = small_val_dataset['raw_labels']

    score = f1_score(y_true, y_pred, average='macro')
    print(f'f1 score: {score}')

    fig = plot_confusion_matrix(y_true, y_pred)
    fig.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))

    # Convert to int label to str
    y_pred = [small_val_dataset.features['raw_labels'].int2str(int(i)) for i in y_pred[:10]]
    y_true = [small_val_dataset.features['raw_labels'].int2str(int(i)) for i in y_true[:10]]

    # Display input, prediction and labels
    inputs = [tokenizer.decode(i, skip_special_tokens=True) for i in small_val_dataset['input_ids'][:10]]
    for input_, label, prediction in zip(inputs, y_true, y_pred):
        lines = textwrap.wrap(f'Inputs:\n{input_}\n', width=100)
        print("\n".join(lines))
        print(f'\nActual label: {label}')
        print(f'Predicted label: {prediction}')
        print("=====================================================================\n")


if __name__ == '__main__':
    main()
