import sys

import pandas as pd
import numpy as np

import datetime

from sklearn.metrics import root_mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from IPython import embed


embeddings = True


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def backtransform(df):
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def train_evaluation(y, yhat):
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))


def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day

    df['td'] = (df['date'] - pd.to_datetime("2013-01-01")).dt.days

    return df


def ewma_prediction(df, group_cols, col, alpha, horizon, suffix=''):
    df.sort_values(["date"], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    df["ewma_{}".format(col + suffix)] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
    return df


def get_events(df):
    for event_date in ['2015-08-07', '2016-08-12', '2017-08-11']:
        for event_days in range(0, 6):
            df.loc[df['date'] == str((pd.to_datetime(event_date) + datetime.timedelta(days=event_days))).split(" ")[0], "Primer Grito de Independencia"] = event_days

    return df


class FF_NN_emb(nn.Module):
    def __init__(self):
        super().__init__()

        if embeddings == True:
            self.family_embedding = nn.Embedding(33, 15)
            self.store_embedding = nn.Embedding(54, 15)
            self.mlp = nn.Sequential(
                nn.Linear(38, 20),
                nn.ReLU(),
                nn.BatchNorm1d(20),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.BatchNorm1d(10),
                nn.Linear(10, 1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(95, 50),
                nn.ReLU(),
                nn.BatchNorm1d(50),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.BatchNorm1d(25),
                nn.Linear(25, 1)
            )

    def forward(self, X):
        if embeddings == True:
            X_store_embed = self.store_embedding(X[:, -2].type(torch.LongTensor).to(device))
            X_family_embed = self.family_embedding(X[:, -1].type(torch.LongTensor).to(device))
            X = torch.cat([X[:, :-2], X_family_embed.squeeze(), X_store_embed.squeeze()], dim=1)
        return self.mlp(X)


def get_model():
    model = FF_NN_emb()
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def fit(epochs, model, optimizer, train_dl, valid_dl):
    loss_func = F.mse_loss

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

        with torch.no_grad():
            valid_loss = sum(loss_func(model(X_mb), y_mb) for X_mb, y_mb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))

    print('Finished training')

    return model


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def nn_prepro(df):
    df['dcoilwtico'].ffill(inplace=True)
    df['ewma_sales_transformed_week'].bfill(inplace=True)
    df.fillna(-999, inplace=True)

    normalized_features = [
        'td',
        'ewma_sales_transformed_week',
        'dayofweek',
        'dayofyear',
        'dayofmonth',
        'onpromotion',
        'dcoilwtico',
        'Primer Grito de Independencia'
    ]
    df[normalized_features] = MinMaxScaler().fit_transform(df[normalized_features])
    print('Finished scaling')

    if embeddings == True:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        df_enc = enc.fit_transform(df[['store_nbr', 'family']])
        df['store_nbr_enc'] = df_enc[: ,0]
        df['family_enc'] = df_enc[:, 1]
        print('Finished ordinal encoding')
    else:
        df_onehot_store = pd.get_dummies(df['store_nbr'], dtype=float)
        df_onehot_family = pd.get_dummies(df['family'], dtype=float)
        df_onehot_family = df_onehot_family.add_suffix('_onehot_family')
        df_onehot_store = df_onehot_store.add_suffix('_onehot_store')
        df = df.merge(df_onehot_store, left_index=True, right_index=True, how='left')
        df = df.merge(df_onehot_family, left_index=True, right_index=True, how='left')
        print('Finished one-hot encoding')

    return df


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    df_train_full = pd.read_csv("train.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

    df_oil = pd.read_csv("oil.csv")
    df_train_full = df_train_full.merge(df_oil, on="date", how="left")

    df_train_full = get_events(df_train_full)

    df_train_full = feature_engineering(df_train_full)

    df_train_full["sales_transformed"] = np.log(1 + df_train_full["sales"])

    ewma_groups = ["store_nbr", "family", "dayofweek"]
    df_train_full = ewma_prediction(df_train_full, ewma_groups, "sales_transformed", 0.15, 1, suffix="_week")

    df_train_full = nn_prepro(df_train_full)

    df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index()
    df_val = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index()

    features = [
        "td",
        "ewma_sales_transformed_week",
        "dayofweek",
        "dayofyear",
        "dayofmonth",
        "onpromotion",
        "dcoilwtico",
        "Primer Grito de Independencia",
    ]

    if embeddings == True:
        features += df_train_full.loc[:, df_train_full.columns.str.endswith('_enc')].columns.tolist()
    else:
        features += df_train_full.loc[:, df_train_full.columns.str.endswith('_onehot_family')].columns.tolist() + df_train_full.loc[:, df_train_full.columns.str.endswith('_onehot_store')].columns.tolist()

    train_target = torch.tensor(df_train["sales_transformed"].astype(np.float32)).unsqueeze(1).to(device)
    train = torch.tensor(df_train[features].values.astype(np.float32)).to(device)
    val_target = torch.tensor(df_val["sales_transformed"].astype(np.float32)).unsqueeze(1).to(device)
    val = torch.tensor(df_val[features].values.astype(np.float32)).to(device)

    train_ds = TensorDataset(train, train_target)
    val_ds = TensorDataset(val, val_target)

    mini_batch_size = 1024
    train_dl, valid_dl = get_data(train_ds, val_ds, mini_batch_size)

    model, optimizer = get_model()
    model = model.to(device)

    epochs = 20
    trained_model = fit(epochs, model, optimizer, train_dl, valid_dl)

    # training data
    df_train["yhat"] = trained_model(train_ds[:][0]).cpu().detach().numpy().flatten()
    df_train["yhat"] = np.clip(df_train["yhat"], 0, None)
    df_train = backtransform(df_train)
    train_evaluation(df_train["sales"], df_train["yhat"])
    # validation data
    df_val["yhat"] = trained_model(val_ds[:][0]).cpu().detach().numpy().flatten()
    df_val["yhat"] = np.clip(df_val["yhat"], 0, None)
    df_val = backtransform(df_val)
    train_evaluation(df_val["sales"], df_val["yhat"])

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
