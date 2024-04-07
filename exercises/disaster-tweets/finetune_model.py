import sys
import os

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
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

    evaluation(np.array(ys), np.array(yhats))

    return np.array(yhats)


def keyword_stack(train_data, test_data, yhat_train, yhat_test):
    X_train = pd.DataFrame({"keyword": train_data["keyword"]})
    X_test = pd.DataFrame({"keyword": test_data["keyword"]})

    y_train = np.array(train_data["label"])

    X_train["yhat_llm"] = yhat_train
    X_test["yhat_llm"] = yhat_test

    features = [
        "keyword",
        "yhat_llm"
    ]

    ml_est = HistGradientBoostingClassifier(
        categorical_features=[
            True,
            False
        ],
        class_weight='balanced',
        random_state=666
    )
    ml_est.fit(X_train[features], y_train)

    yhat_train = ml_est.predict(X_train[features])
    yhat_test = ml_est.predict(X_test[features])

    return yhat_train, yhat_test


def main(args):
    np.random.seed(42)
    torch.manual_seed(666)

    df_train_full = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    df_train_full.fillna("", inplace=True)
    df_test.fillna("", inplace=True)

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    train_data = Dataset.from_pandas(df_train)
    train_data_full = Dataset.from_pandas(df_train_full)
    val_data = Dataset.from_pandas(df_val)
    test_data = Dataset.from_pandas(df_test)

    train_data = train_data.rename_column("target", "label")
    train_data_full = train_data_full.rename_column("target", "label")
    val_data = val_data.rename_column("target", "label")
    train_data_full = train_data_full.remove_columns(['location'])
    test_data = test_data.remove_columns(['location'])
    train_data = train_data.remove_columns(['location', '__index_level_0__'])
    val_data = val_data.remove_columns(['location', '__index_level_0__'])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=access_token)

    # validation
    finetuning(train_data, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs")

    yhat_train = llm_predict(train_data, tokenizer, model)
    yhat_val = llm_predict(val_data, tokenizer, model)

    yhat_train, yhat_val = keyword_stack(train_data, val_data, yhat_train, yhat_val)
    evaluation(np.array(train_data["label"]), yhat_train)
    evaluation(np.array(val_data["label"]), yhat_val)

    # test
    finetuning(train_data_full, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs")

    yhat_train_full = llm_predict(train_data_full, tokenizer, model)

    yhats = []
    for sample in test_data:
        input_ids = tokenizer(sample["text"], return_tensors="pt")
        outputs = F.softmax(model(**input_ids)[0], dim=1).detach().numpy()
        yhat = np.argmax(outputs, axis=1)
        yhats.append(yhat[0])
    yhat_test = np.array(yhats)

    yhat_train_full, yhat_test = keyword_stack(train_data_full, test_data, yhat_train_full, yhat_test)
    evaluation(np.array(train_data_full["label"]), yhat_train_full)

    pd.concat([df_test["id"], pd.Series(yhat_test, name="target")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
