import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from IPython import embed


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=25, kernel_size=5),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(in_channels=25, out_channels=40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(750, 300),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, 10),
            # nn.Softmax(dim=1)

        # LeNet
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        #     nn.Sigmoid(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        #     nn.Sigmoid(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(400, 120),
        #     nn.Sigmoid(),
        #     nn.Linear(120, 84),
        #     nn.Sigmoid(),
        #     nn.Linear(84, 10)
        )

    def forward(self, X):
        X = self.cnn(X)
        return X


def fit(epochs, model, optimizer, train_dl, valid_dl=None):
    loss_func = nn.CrossEntropyLoss()

    # loop over epochs
    for epoch in range(epochs):
        model.train()

        # loop over mini-batches
        for X_mb, y_mb in train_dl:
            y_hat = model(X_mb)

            loss = loss_func(y_hat, y_mb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

        model.eval()

        if valid_dl:
            with torch.no_grad():
                valid_loss = sum(loss_func(model(X_mb), y_mb) for X_mb, y_mb in valid_dl)

            print(epoch, valid_loss / len(valid_dl))

    print('Finished training')

    return model


def get_model():
    model = CNN()
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def main(args):
    torch.manual_seed(666)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    df = pd.read_csv("train.csv")
    y = df["label"]
    X = df.drop(columns="label")

    # for i in range(20, 50):
    #     example_image = np.asarray(X.iloc[i]).reshape(28, 28)
    #     plt.imshow(example_image, cmap='gray', vmin=0, vmax=255)
    #     plt.savefig("plots/digit_train_{}.png".format(i))
    #     plt.clf()

    df_test = pd.read_csv("test.csv")

    # for i in range(30):
    #     example_image = np.asarray(df_test.iloc[i]).reshape(28, 28)
    #     plt.imshow(example_image, cmap='gray', vmin=0, vmax=255)
    #     plt.savefig("plots/digit_test_{}.png".format(i))
    #     plt.clf()

    X = np.asarray(X).reshape(-1, 28, 28)
    y = np.asarray(y)

    X_test = np.asarray(df_test).reshape(-1, 28, 28)

    # mini_batch_size = 1024
    mini_batch_size = 512
    # mini_batch_size = 256

    epochs = 50

    model, optimizer = get_model()
    model = model.to(device)

    validation_run = True
    if validation_run:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        target_train = torch.tensor(F.one_hot(torch.tensor(y_train)), dtype=torch.float32).to(device)
        train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        train_ds = TensorDataset(train, target_train)

        target_test = torch.tensor(F.one_hot(torch.tensor(y_test)), dtype=torch.float32).to(device)
        test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        test_ds = TensorDataset(test, target_test)

        train_dl, valid_dl = get_data(train_ds, test_ds, mini_batch_size)

        trained_model = fit(epochs, model, optimizer, train_dl, valid_dl)

        train_preds = trained_model(train_ds[:][0]).cpu()
        train_preds = F.softmax(train_preds, dim=1).detach().numpy()
        yhat = np.argmax(train_preds, axis=1)
        print('accuracy: ', accuracy_score(y_train, yhat))

        test_preds = trained_model(test_ds[:][0]).cpu()
        test_preds = F.softmax(test_preds, dim=1).detach().numpy()
        yhat = np.argmax(test_preds, axis=1)
        print('accuracy: ', accuracy_score(y_test, yhat))
    else:
        target_train = torch.tensor(F.one_hot(torch.tensor(y)), dtype=torch.float32).to(device)
        train = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        train_ds = TensorDataset(train, target_train)

        train_dl = DataLoader(train_ds, batch_size=mini_batch_size, shuffle=True)

        trained_model = fit(epochs, model, optimizer, train_dl)

        train_preds = trained_model(train_ds[:][0]).cpu()
        train_preds = F.softmax(train_preds, dim=1).detach().numpy()
        yhat = np.argmax(train_preds, axis=1)
        print('accuracy: ', accuracy_score(y, yhat))

        test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        test_ds = TensorDataset(test)

        test_preds = trained_model(test_ds[:][0]).cpu()
        test_preds = F.softmax(test_preds, dim=1).detach().numpy()
        yhat = np.argmax(test_preds, axis=1)
        df_test["ImageId"] = df_test.index + 1
        pd.concat([df_test["ImageId"], pd.Series(yhat, name="Label")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
