import sys
import os

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification
from datasets import Dataset

import evaluate

from IPython import embed


access_token = os.environ['HF_TOKEN']


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def finetuning(train_data, val_data, tokenizer):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = 2,
    )
    print(model)

    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    # only 0.75 F1 score without finetuning pre-trained layers (0.82 with)

    train_data = train_data.map(lambda samples: tokenizer(samples["text"], max_length=300, padding="max_length", truncation=True), batched=True)
    val_data = val_data.map(lambda samples: tokenizer(samples["text"], max_length=300, padding="max_length", truncation=True), batched=True)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
            output_dir="outputs",
            num_train_epochs=2,
            evaluation_strategy="epoch"
        )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        compute_metrics=compute_metrics
    )
    trainer.train()

    model.save_pretrained("outputs")


def llm_predict(data, tokenizer, model):
    yhats = []
    ys = []
    for sample in data:
        input_ids = tokenizer(sample["text"], return_tensors="pt")
        outputs = F.softmax(model(**input_ids)[0], dim=1).detach().numpy()
        yhat = np.argmax(outputs, axis=1)
        yhats.append(yhat[0])
        y = sample["label"]
        ys.append(y)

    return np.array(ys), np.array(yhats)


def main(args):
    np.random.seed(42)
    torch.manual_seed(666)

    df_train_full = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    df_train_full = df_train_full[["text", "target"]]
    df_test = df_test[["text", "id"]]

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    train_data = Dataset.from_pandas(df_train)
    train_data_full = Dataset.from_pandas(df_train_full)
    val_data = Dataset.from_pandas(df_val)
    test_data = Dataset.from_pandas(df_test)

    train_data = train_data.rename_column("target", "label")
    train_data_full = train_data_full.rename_column("target", "label")
    val_data = val_data.rename_column("target", "label")
    train_data = train_data.remove_columns(['__index_level_0__'])
    val_data = val_data.remove_columns(['__index_level_0__'])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=access_token)

    # validation
    finetuning(train_data, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs")

    y_train, yhat_train_llm = llm_predict(train_data, tokenizer, model)
    evaluation(y_train, yhat_train_llm)

    y_val, yhat_val_llm = llm_predict(val_data, tokenizer, model)
    evaluation(y_val, yhat_val_llm)

    # test
    finetuning(train_data_full, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs")

    y_train_full, yhat_train_full_llm = llm_predict(train_data_full, tokenizer, model)
    evaluation(y_train_full, yhat_train_full_llm)

    yhats = []
    for sample in test_data:
        input_ids = tokenizer(sample["text"], return_tensors="pt")
        outputs = F.softmax(model(**input_ids)[0], dim=1).detach().numpy()
        yhat = np.argmax(outputs, axis=1)
        yhats.append(yhat[0])
    yhat_test = np.array(yhats)

    pd.concat([df_test["id"], pd.Series(yhat_test, name="target")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
