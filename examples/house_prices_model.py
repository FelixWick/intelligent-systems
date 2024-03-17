import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

from IPython import embed


def target_correlation_plots(df_train, yhat):
    df_train["yhat"] = yhat
    for feature in df_train.columns:
        if df_train[feature].dtype != 'O':
            df_train = df_train.loc[df_train[feature].notnull()]
            sns.regplot(x=df_train[feature], y=df_train["SalePrice"], x_bins=100, fit_reg=None)
            sns.regplot(x=df_train[feature], y=df_train["yhat"], x_bins=100, fit_reg=None, color="red")
            plt.savefig("plots/target_profile_{}.png".format(feature))
            plt.clf()


def training(X, y):
    categorical_flags = []
    for col in X.columns:
        if X[col].dtype == 'O':
            categorical_flag = True
        else:
            categorical_flag = False
        categorical_flags.append(categorical_flag)

    ml_est = HistGradientBoostingRegressor(
        categorical_features=categorical_flags
    )
    ml_est.fit(X, y)

    return ml_est


def train_evaluation(ml_est, X, y):
    yhat = np.exp(ml_est.predict(X))
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))

    return yhat


def main(args):
    df_train_full = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    y = df_train_full["SalePrice"]
    X = df_train_full.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    ml_est = training(X_train, np.log(y_train))
    train_evaluation(ml_est, X_test, y_test)

    ml_est = training(X, np.log(y))
    yhat_train_full = train_evaluation(ml_est, X, y)

    target_correlation_plots(df_train_full, yhat_train_full)

    yhat = np.exp(ml_est.predict(df_test.drop(columns=["Id"])))
    pd.concat([df_test["Id"], pd.Series(yhat, name="SalePrice")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
