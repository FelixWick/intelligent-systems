import sys

import torch.nn.functional as F

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification
from datasets import load_dataset

import evaluate

from IPython import embed


access_token='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'

def train_evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def finetuning(train_data, val_data, tokenizer):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels = 2,
    )
    print(model)

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    train_data = train_data.map(lambda samples: tokenizer(samples["text"], padding="max_length", truncation=True), batched=True)
    val_data = val_data.map(lambda samples: tokenizer(samples["text"], padding="max_length", truncation=True), batched=True)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
            output_dir="outputs",
            num_train_epochs=2,
            # label_names="target",
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


def main(args):
    np.random.seed(42)

    # data = load_dataset("mehdiiraqui/twitter_disaster")

    data = load_dataset("MAdAiLab/twitter_disaster")
    train_data = data["train"]
    val_data = data["validation"]
    test_data = data["test"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", token=access_token)

    finetuning(train_data, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs")

    yhats = []
    ys = []
    for sample in test_data:
        input_ids = tokenizer(sample["text"], return_tensors="pt")
        outputs = F.softmax(model(**input_ids)[0], dim=1).detach().numpy()
        yhat = np.argmax(outputs, axis=1)
        yhats.append(yhat[0])
        y = sample["label"]
        ys.append(y)

    train_evaluation(np.array(ys), np.array(yhats))

    # input_ids = tokenizer(test_data["text"], return_tensors="pt", padding="max_length", truncation=True)
    # outputs = F.softmax(model(**input_ids)[0], dim=1).detach().numpy()
    # yhat = np.argmax(outputs, axis=1)
    # y = test_data["label"].detach().numpy()
    # train_evaluation(y, yhat)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
