import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from IPython import embed


def target_correlation_plots(df_train, yhat):
    df_train["yhat"] = yhat
    for feature in ["number_in_group", "group_size", "single_group", "num", "cabin_size", "single_cabin", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df_train = df_train.loc[df_train[feature].notnull()]
        df_train[feature] = np.where(df_train[feature].dtype == object, df_train[feature].astype(int), df_train[feature])
        sns.regplot(x=df_train[feature], y=df_train["Transported"], x_bins=100, fit_reg=None)
        sns.regplot(x=df_train[feature], y=df_train["yhat"], x_bins=100, fit_reg=None, color="red")
        plt.savefig("plots/target_profile_{}.png".format(feature))
        plt.clf()


def training(df, features):
    y = df["Transported"]
    X = df[features]

    ml_est = HistGradientBoostingClassifier(
        categorical_features=[
            False,
            True,
            # False,
            True,
            True,
            True,
            False,
            True,
            True,
            # False,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False
        ]
    )
    ml_est.fit(X, y)

    return ml_est, X, y


def train_evaluation(ml_est, X, y):
    yhat = ml_est.predict(X)
    print('accuracy: ', accuracy_score(y, yhat))
    print('mean(y): ', np.mean(y))

    return yhat


def feature_engineering(df):
    df_PassengerId = df["PassengerId"].str.split("_", expand=True)
    df["group"] = df_PassengerId[0].astype(int)
    df["number_in_group"] = df_PassengerId[1].astype(int)
    df["group_size"] = df.groupby("group")["number_in_group"].transform("count")
    df["single_group"] = np.where(df["group_size"] == 1, True, False)

    df_Name = df["Name"].str.split(" ", expand=True)
    df["first_name"] = df_Name[0]
    df["last_name"] = df_Name[1]

    df_Cabin = df["Cabin"].str.split("/", expand=True)
    df["deck"] = df_Cabin[0]
    df["num"] = df_Cabin[1]
    df["side"] = df_Cabin[2]
    df["cabin_size"] = df.groupby("Cabin")["num"].transform("count")
    df["single_cabin"] = np.where(df["cabin_size"] == 1, True, False)

    return df


def main(args):
    df_train_full = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    df_train_full = feature_engineering(df_train_full)
    df_test = feature_engineering(df_test)

    np.random.seed(666)
    validation_groups = np.random.randint(0, len(df_train_full["group"].unique()), size=1000)
    df_val = df_train_full.loc[df_train_full["group"].isin(validation_groups)]
    df_train = df_train_full.loc[~df_train_full["group"].isin(validation_groups)]

    features = [
        "number_in_group",
        "single_group",
        # "group_size",
        "HomePlanet",
        "CryoSleep",
        "deck",
        "num",
        "side",
        "single_cabin",
        # "cabin_size",
        "Destination",
        "Age",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    ml_est, _, _ = training(df_train, features)
    train_evaluation(ml_est, df_val[features], df_val["Transported"])

    ml_est, X, y = training(df_train_full, features)
    yhat_train_full = train_evaluation(ml_est, X, y)

    target_correlation_plots(df_train_full, yhat_train_full)

    yhat = ml_est.predict(df_test[features])
    pd.concat([df_test["PassengerId"], pd.Series(yhat, name="Transported")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
