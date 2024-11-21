import os
import random
import evaluate
import mlflow
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline
)

from constants import HUGGING_FACE_MODEL, PRICE_CLASSES_AMOUNT


# disable WandB defaults
os.environ['WANDB_DISABLED'] = 'true'

# a seed for reproducibility
SEED = 42
# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# check for GPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device available:', device) 

df_train = pd.read_csv('../data/train_data.csv')
df_test = pd.read_csv('../data/test_data.csv')

df_train, df_val = train_test_split(
    df_train, test_size=0.1, random_state=42
)

df_train = df_train[['text', 'price_class']]
df_test = df_test[['text', 'price_class']]
df_val = df_val[['text', 'price_class']]

df_train['text'] = df_train['text'].astype(str)
df_test['text'] = df_test['text'].astype(str)
df_val['text'] = df_val['text'].astype(str)

train_dataset = Dataset.from_pandas(df_train).remove_columns('__index_level_0__')
test_dataset = Dataset.from_pandas(df_test)
val_dataset = Dataset.from_pandas(df_val).remove_columns('__index_level_0__')

full_dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'val': val_dataset
})

tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE_MODEL)
# data collator for dynamic padding as per batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = BertForSequenceClassification.from_pretrained(
    HUGGING_FACE_MODEL,
    num_labels=PRICE_CLASSES_AMOUNT
)

# define a tokenize function
def Tokenize_function(example):
    return tokenizer(example['text'], truncation=True)

tokenized_data = full_dataset.map(Tokenize_function, batched=True)

tokenized_data = tokenized_data.rename_column('price_class','labels')
tokenized_data.with_format('pt')

# use the pre-built metrics 
def compute_metrics(eval_preds):
    f1_metric = evaluate.load('f1')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='macro'
    )

def calculate_metrics(y_pred, y_test, average):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    f1 = f1_score(y_test, y_pred, average=average)

    return accuracy, precision, recall, f1

y_test = test_dataset['price_class']

mlflow.set_tracking_uri('../mlruns')
mlflow.set_experiment(f'Text classificator ({PRICE_CLASSES_AMOUNT})')

with mlflow.start_run():
    training_params = {
        'output_dir': 'bert-finetuning',
        'eval_strategy': 'epoch',
        'num_train_epochs': 3,
        'learning_rate': 5e-5,
        'weight_decay': 0.005,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'report_to': 'none',
    }

    model_config = {'batch_size': 8}

    training_args = TrainingArguments(**training_params)
    
    mlflow.log_params(training_params)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['val'],
        data_collator=data_collator,
        processing_class =tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    tuned_pipeline = pipeline(
        task='text-classification',
        model=trainer.model,
        batch_size=8,
        tokenizer=tokenizer,
        device='cpu',
    )
    predictions = trainer.predict(tokenized_data['test'])
    y_pred = np.argmax(predictions.predictions, axis=1) 

    accuracy, precision, recall, f1 = calculate_metrics(y_pred, y_test, 'macro')
    
    print('Rubert tiny2 model')
    print(f'  Accuracy: {accuracy}')
    print(f'  Precision: {precision}')
    print(f'  Recall: {recall}')
    print(f'  F1: {f1}')

    mlflow.log_metric('Accuracy', accuracy)
    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F1', f1)

    
    # Логирование модели с примером входных данных
    mlflow.transformers.log_model(
        transformers_model=tuned_pipeline,
        artifact_path='bert-finetuning',
        model_config=model_config,
    )

df = pd.read_csv('../data/ml_data.csv')

mlflow.set_tracking_uri('../mlruns')
loaded_model = mlflow.pyfunc.load_model('runs:/798e3dacc2794a3fa23cc7e9cc2ee33f/bert-finetuning')

prob_preds = loaded_model.predict(df['text'].astype(str).values)
df['price_class_pred'] = prob_preds['label'].apply(lambda x: int(x[-1]))

df.drop('price_class', axis=1, inplace=True)
df.to_csv('../data/ml_data.csv', index=False)

df_train = pd.read_csv('../data/train_data.csv')
df_test = pd.read_csv('../data/test_data.csv')

prob_preds_train = loaded_model.predict(df_train['text'].astype(str).values)
prob_preds_test = loaded_model.predict(df_test['text'].astype(str).values)

df_train.drop('price_class', axis=1, inplace=True)
df_test.drop('price_class', axis=1, inplace=True)

df_train['price_class_pred'] = prob_preds_train['label'].apply(lambda x: int(x[-1]))
df_test['price_class_pred'] = prob_preds_test['label'].apply(lambda x: int(x[-1]))

df_train.to_csv('../data/train_data.csv', index=False)
df_test.to_csv('../data/test_data.csv', index=False)
