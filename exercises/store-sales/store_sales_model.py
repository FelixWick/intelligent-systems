import sys

import pandas as pd
import numpy as np

import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_log_error

from IPython import embed


def plot_timeseries(df, include_preds=False):
    if include_preds:
        ts = df.groupby(['date'])[['sales', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['sales'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['sales'].plot(style='r', label="sales")
    if include_preds:
        ts['yhat'].plot(style='b-.', label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("plots/ts.png")
    plt.clf()


def target_correlation_plots(df_train):
    for feature in ["ewma_sales_transformed_week", "td", "onpromotion", "dayofweek", "dayofyear", "dayofmonth", "dcoilwtico", "Primer Grito de Independencia"]:
    # for feature in ["Primer Grito de Independencia"]:
        sns.regplot(x=df_train[feature], y=df_train["sales"], x_bins=365, fit_reg=None)
        sns.regplot(x=df_train[feature], y=df_train["yhat"], x_bins=365, fit_reg=None, color="red")
        plt.savefig("plots/target_profile_{}.png".format(feature))
        plt.clf()


def prediction(df, ml_est, features):
    yhat = ml_est.predict(df[features])
    df["yhat"] = yhat
    df["yhat"] = np.clip(df["yhat"], 0, None)
    return df


def backtransform(df):
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def training(X, y):
    ml_est = HistGradientBoostingRegressor(
        categorical_features=[
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        random_state=666,
        # interaction_cst=[{1}]
    )
    ml_est.fit(X, y)

    return ml_est


def train_evaluation(y, yhat):
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))


def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['midofmonthpayout'] = np.where(df['dayofmonth'] == 16, 1, 0)
    df['endofmonthpayout'] = np.where(df['dayofmonth'] == 31, 1, 0)

    df['td'] = (df['date'] - pd.to_datetime("2013-01-01")).dt.days

    return df


def ewma_prediction(df, group_cols, col, alpha, horizon, suffix=''):
    df.sort_values(["date"], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    df["ewma_{}".format(col + suffix)] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
    return df


def ewma_merge(df_test, df_train, ewma_col, group_cols):
    def get_latest_ewmas(df):
        return df.loc[df["date"] == df["date"].max(), ewma_col]

    df_train_latest_ewma = df_train[["date", ewma_col] + group_cols].groupby(group_cols).apply(get_latest_ewmas).reset_index()

    df_test = df_test.merge(df_train_latest_ewma[[ewma_col] + group_cols], on=group_cols, how="left")

    return df_test


# def residual_correction(df):
#     df["correction_factor"] = df["ewma_sales_transformed"] / df["ewma_yhat"]
#     # df["correction_factor"] = np.clip(df["correction_factor"], 0.5, 2.)
#     df["correction_factor"] = np.where((df["correction_factor"] > 0.5) & (df["correction_factor"] < 2.), df["correction_factor"], 1.)
#     mask = df["correction_factor"].notna()
#     df.loc[mask, "yhat"] = df.loc[mask, "correction_factor"] * df.loc[mask, 'yhat']
#     return df


def get_events(df):
    # for event_date in ['2013-08-10', '2014-08-10', '2015-08-10', '2016-08-12', '2017-08-11']:
    for event_date in ['2015-08-07', '2016-08-12', '2017-08-11']:
        for event_days in range(0, 6):
            df.loc[df['date'] == str((pd.to_datetime(event_date) + datetime.timedelta(days=event_days))).split(" ")[0], "Primer Grito de Independencia"] = event_days

    return df


def main(args):
    df_train_full = pd.read_csv("train.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]
    # plot_timeseries(df_train_full[(df_train_full["date"] >= "2016-08-01") & (df_train_full["date"] <= "2016-08-31")])
    # plot_timeseries(df_train_full[df_train_full["date"] >= "2017-01-01"])
    df_test = pd.read_csv("test.csv")

    df_oil = pd.read_csv("oil.csv")
    df_train_full = df_train_full.merge(df_oil, on="date", how="left")
    df_test = df_test.merge(df_oil, on="date", how="left")

    df_stores = pd.read_csv("stores.csv")
    df_train_full = df_train_full.merge(df_stores, on="store_nbr", how="left")
    df_train_full.rename(columns={"type": "store_type"}, inplace=True)
    df_test = df_test.merge(df_stores, on="store_nbr", how="left")
    df_test.rename(columns={"type": "store_type"}, inplace=True)

    df_events = pd.read_csv("holidays_events.csv")
    df_events_national = df_events[df_events["locale"] == "National"]
    df_train_full = df_train_full.merge(df_events_national[["date", "type", "description", "transferred"]], on="date", how="left")
    df_train_full.rename(columns={"type": "national_event_type", "description": "national_description", "transferred": "national_transferred"}, inplace=True)
    df_test = df_test.merge(df_events_national[["date", "type", "description", "transferred"]], on="date", how="left")
    df_test.rename(columns={"type": "national_event_type", "description": "national_description", "transferred": "national_transferred"}, inplace=True)
    df_events_regional = df_events[df_events["locale"] == "Regional"]
    df_train_full = df_train_full.merge(df_events_regional[["date", "type", "description", "transferred", "locale_name"]], left_on=["date", "state"], right_on=["date", "locale_name"], how="left")
    df_train_full.rename(columns={"type": "regional_event_type", "description": "regional_description", "transferred": "regional_transferred"}, inplace=True)
    df_test = df_test.merge(df_events_regional[["date", "type", "description", "transferred", "locale_name"]], left_on=["date", "state"], right_on=["date", "locale_name"], how="left")
    df_test.rename(columns={"type": "regional_event_type", "description": "regional_description", "transferred": "regional_transferred"}, inplace=True)

    df_train_full = get_events(df_train_full)
    df_test = get_events(df_test)

    df_train_full = feature_engineering(df_train_full)
    df_test = feature_engineering(df_test)

    df_val = df_train_full[df_train_full["date"] >= "2017-07-31"]
    # df_val = df_train_full[(df_train_full["date"] >= "2016-08-16") & (df_train_full["date"] <= "2016-08-31")]

    df_train_full["sales_transformed"] = np.log(1 + df_train_full["sales"])

    ewma_groups = ["store_nbr", "family", "dayofweek"]
    df_train_full = ewma_prediction(df_train_full, ewma_groups, "sales_transformed", 0.15, 1, suffix="_week")

    df_train = df_train_full[df_train_full["date"] <= "2017-07-30"]
    # df_train = df_train_full[df_train_full["date"] <= "2016-08-15"]

    features = [
        "td",
        "ewma_sales_transformed_week",
        "store_nbr",
        "family",
        "cluster",
        "store_type",
        "dayofweek",
        "dayofyear",
        "dayofmonth",
        "midofmonthpayout",
        "onpromotion",
        "dcoilwtico",
        "national_event_type",
        "national_description",
        "regional_description",
        "Primer Grito de Independencia",
    ]

    # validation training
    ml_est = training(df_train[features], df_train["sales_transformed"])
    df_train = ewma_prediction(df_train, ewma_groups, "sales_transformed", 0.15, 0, suffix="_week")
    df_val = ewma_merge(df_val, df_train, "ewma_sales_transformed_week", ewma_groups)
    prediction(df_val, ml_est, features)
    df_val = backtransform(df_val)
    train_evaluation(df_val["sales"], df_val["yhat"])
    plot_timeseries(df_val, True)

    # full training
    ml_est = training(df_train_full[features], df_train_full["sales_transformed"])

    # in-sample test
    prediction(df_train_full, ml_est, features)
    df_train_full = backtransform(df_train_full)
    train_evaluation(df_train_full["sales"], df_train_full["yhat"])
    target_correlation_plots(df_train_full)

    # test data
    df_train_full = ewma_prediction(df_train_full, ewma_groups, "sales_transformed", 0.15, 0, suffix="_week")
    df_test = ewma_merge(df_test, df_train_full, "ewma_sales_transformed_week", ewma_groups)
    prediction(df_test, ml_est, features)
    df_test = backtransform(df_test)
    pd.concat([df_test["id"], df_test["yhat"]], axis=1).rename(columns={"yhat": "sales"}).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
