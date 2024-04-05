import sys

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, DistilBertModel

from IPython import embed


access_token='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'


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


class DistilBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        print(self.base_model)

        D_in, D_out = 769, 2
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_in),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(D_in, D_out)
            )

    def forward(self, input_ids, attention_mask, train_keyword):
        base_output = self.base_model(input_ids, attention_mask)
        hidden_state = base_output[0]
        pooled_output = hidden_state[:, 0]

        X = torch.cat([pooled_output, train_keyword.unsqueeze(1)], dim=1)

        return self.classifier(X)


def fit(epochs, model, optimizer, train_dl):
    loss_func = nn.CrossEntropyLoss()

    # loop over epochs
    for epoch in range(epochs):
        model.train()

        # loop over mini-batches
        for input_ids, attention_mask, labels, keyword in train_dl:
            y_hat = model(input_ids, attention_mask, keyword)

            loss = loss_func(y_hat, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # print(loss.item())

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    print('Finished training')

    return model


def tokenization(tokenizer, text_data):
    encoded_corpus = tokenizer(
        text_data,
        max_length=157,
        padding="max_length",
        truncation=True,
        return_attention_mask=True
        )
    input_ids = torch.tensor(encoded_corpus['input_ids'])
    attention_mask = torch.tensor(encoded_corpus['attention_mask'])

    return input_ids, attention_mask


def predict(model, dataloader):
    model.eval()

    yhat = []
    y = []
    for input_ids, attention_mask, labels, keyword in dataloader:
        with torch.no_grad():
            output = F.softmax(model(input_ids, attention_mask, keyword), dim=1).cpu().detach().numpy()
            output = np.argmax(output, axis=1).tolist()
            yhat += output

        y += labels.tolist()

    return yhat, y


def main(args):
    np.random.seed(42)
    torch.manual_seed(666)

    df_train_full = pd.read_csv("../train.csv")
    # df_test = pd.read_csv("../test.csv")

    df_train_full.fillna("", inplace=True)
    # df_test.fillna("", inplace=True)

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", token=access_token)

    # finetuning
    input_ids_train, attention_mask_train = tokenization(tokenizer, df_train["text"].tolist())

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    train_keyword = pd.DataFrame({"keyword": df_train["keyword"]})
    train_keyword = torch.tensor(enc.fit_transform(train_keyword)).view(-1).float().to(device)

    model = DistilBertClassifier()
    print(model)

    for param in model.base_model.parameters():
        param.requires_grad = False

    labels = torch.tensor(df_train["target"].tolist()).to(device)

    train_dataset = TensorDataset(input_ids_train.to(device), attention_mask_train.to(device), labels, train_keyword)
    train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters())
    epochs = 2
    trained_model = fit(epochs, model.to(device), optimizer, train_dl)

    torch.save(trained_model, "outputs/model.pth")

    # inference
    input_ids_test, attention_mask_test = tokenization(tokenizer, df_val["text"].tolist())

    test_keyword = pd.DataFrame({"keyword": df_val["keyword"]})
    test_keyword = torch.tensor(enc.transform(test_keyword)).view(-1).float().to(device)

    labels_test = torch.tensor(df_val["target"].tolist()).to(device)

    test_dataset = TensorDataset(input_ids_test.to(device), attention_mask_test.to(device), labels_test, test_keyword)
    test_dl = DataLoader(test_dataset, batch_size=128)

    inference_model = torch.load("outputs/model.pth")

    yhat, y = predict(inference_model, test_dl)

    evaluation(y, yhat)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
