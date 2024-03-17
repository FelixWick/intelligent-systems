import sys

import requests

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

from IPython import embed


# ####
# Here are some examples:

# """
#     for i in range(len(most_important_keywords)):
#         examples_string += """Inquiry: {}
# This tweet can be characterized by the keyword {}.
# {}
# """.format(example_text[i], example_keyword[i], example_target[i])
#     examples_string += """####

# """

def build_examples_string(X_train, most_important_keywords):
    df_one_representative = X_train[X_train["keyword"].isin(most_important_keywords)]
    df_one_representative["mean_target"] = df_one_representative.groupby("keyword")["target"].transform("mean")
    df_one_representative = df_one_representative[((df_one_representative["mean_target"] >= 0.5) & (df_one_representative["target"] == 1)) | ((df_one_representative["mean_target"] < 0.5) & (df_one_representative["target"] == 0))]
    df_one_representative = df_one_representative.groupby("keyword").sample(n=1, random_state=42)

    example_text = df_one_representative["text"].values
    # example_keyword = df_one_representative["keyword"].values
    # example_target = df_one_representative["target"].values
    # example_target = [0.99, 0.18, 0.03, 0.72, 0.13, 0.01, 0.68, 0.95, 0.82, 0.75, 0.65, 0.55, 0.15, 0.05, 0.12, 0.92, 0.97, 0.03, 0.01, 0.15, 0.03, 0.2, 0.02, 0.03, 0.08]
    example_target = [0.96, 0.17, 0.01, 0.76, 0.09, 0.19, 0.69, 0.98, 0.79, 0.52, 0.66, 0.54, 0.16, 0.05, 0.07, 0.82, 0.87, 0.03, 0.02, 0.13, 0.06, 0.25, 0.04, 0.08, 0.21]
    examples_string = """

####
Here are some examples:

"""
    for i in range(len(most_important_keywords)):
        examples_string += """Inquiry: {}
{}
""".format(example_text[i], example_target[i])
    examples_string += """####

"""

    print(examples_string)

    return examples_string


def build_simple_examples_string():
    example_string = """

####
Here is a disaster example:

Inquiry: Heavy accident on Interstate 10.
0.95

Here is a non-disaster example:

Inquiry: Good morning everybody.
0.05
####

"""

    print(example_string)

    return example_string


# <<<
# Inquiry: {}
# This tweet can be characterized by the keyword {}.
# This tweet was sent from the location {}.
# >>>""".format(inquiry, keyword, location)

    # instruction = """You are a social media analyst. Your task is to classify a tweet after <<<>>> to be either about a disaster or about something else. Return '1' if you think the provided tweet is actually about a disaster. Return '0' if it's really about something else.
# You will only respond with '1' or '0'. Do not provide any explanations or notes."""

def model(X_test, examples_string="", train_target_mean=0.5):
    yhat = []

    instruction = """You are a social media analyst. Your task is to classify a tweet after <<<>>> to be either about a disaster or about something else. Return a numerical value between 0 and 1 for the probability that the provided tweet is actually about a disaster.
You will only respond with the numerical value for the disaster probability. Do not provide any explanations or notes."""

    for i in range(len(X_test)):
        inquiry = X_test["text"].iloc[i]
        # keyword = X_test["keyword"].iloc[i]
        # location = X_test["location"].iloc[i]

        prompt = examples_string + """

<<<
Inquiry: {}
>>>""".format(inquiry)

        json_data = {
            "model": "mistral",
            "system": instruction,
            "prompt": prompt,
            "stream": False,
            # "options": {"num_predict": 2, "seed": 42, "temperature": 0},
            "options": {"num_predict": 5, "seed": 42, "temperature": 0},
        }

        response = requests.post('http://localhost:11434/api/generate', json=json_data)
        yhat_item = response.json()["response"]
        # print(yhat_item)
        yhat_item = yhat_item.strip(' .')
        # if yhat_item not in ['0', '1']:
        try:
            yhat_item = float(yhat_item)
        except ValueError:
            # yhat_item = 0
            yhat_item = train_target_mean
            print("blabla")
        # yhat_item = int(yhat_item)
        # print("final:{}".format(yhat_item))
        yhat.append(yhat_item)

    return np.array(yhat)


def llm_run():
    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    train_target_mean = df_train["target"].mean()

    df_train.fillna("", inplace=True)
    df_test.fillna("", inplace=True)

    most_important_keywords = df_train["keyword"].value_counts()[:25].keys().values

    y = df_train["target"]
    print('mean(y): ', np.mean(y))
    X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=666)
    X_train["target"] = y_train

    examples_string = build_examples_string(X_train, most_important_keywords)
    # examples_string = build_simple_examples_string()
    # examples_string = ""

    validation_run = True
    fast_test = False
    stacking_run = True
    if validation_run:
        if fast_test:
            test_samples = np.random.choice(range(len(X_test)), 100, replace=False)
            yhat = model(X_test.iloc[test_samples], examples_string, train_target_mean)
            yhat = np.where(yhat > train_target_mean, 1, 0)
            print('accuracy: ', accuracy_score(np.asarray(y_test.iloc[test_samples]), yhat))
            print('f1_score: ', f1_score(np.asarray(y_test.iloc[test_samples]), yhat))
            print('mean(y_test): ', np.mean(y_test.iloc[test_samples]))
        else:
            yhat = model(X_test, examples_string, train_target_mean)
            yhat = np.where(yhat > train_target_mean, 1, 0)
            print('accuracy: ', accuracy_score(np.asarray(y_test), yhat))
            print('f1_score: ', f1_score(np.asarray(y_test), yhat))
            print('mean(y_test): ', np.mean(y_test))
            # X_test["yhat"] = yhat
            # X_test["target"] = y_test
            # X_test.to_csv("submission_test.csv", index=False)
        print('mean(yhat): ', np.mean(yhat))
    else:
        yhat = model(df_test, examples_string, train_target_mean)
        # pd.concat([df_test["id"], pd.Series(yhat, name="target")], axis=1).to_csv("submission.csv", index=False)
        if stacking_run:
            df_test["yhat_llm"] = yhat
            df_test.to_csv("stacking_test_prob.csv", index=False)
            yhat = model(df_train, examples_string, train_target_mean)
            df_train["yhat_llm"] = yhat
            df_train.to_csv("stacking_train_prob.csv", index=False)

    embed()


def training(X, y):
    ml_est = HistGradientBoostingClassifier(
        categorical_features=[
            True,
            False
        ],
        class_weight='balanced',
        random_state=666
    )
    ml_est.fit(X, y)

    return ml_est


def train_evaluation(ml_est, X, y):
    yhat = ml_est.predict(X)
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def numerical_model():
    # df_train_full = pd.read_csv("stacking_train.csv")
    # df_test = pd.read_csv("stacking_test.csv")
    df_train_full = pd.read_csv("stacking_train_prob.csv")
    df_test = pd.read_csv("stacking_test_prob.csv")
    # df_train_full = pd.read_csv("train.csv")
    # df_test = pd.read_csv("test.csv")

    y = df_train_full["target"]
    X_train, X_test, y_train, y_test = train_test_split(df_train_full, y, test_size=0.2, random_state=666)

    features = [
        "keyword",
        "yhat_llm"
    ]

    ml_est = training(X_train[features], y_train)
    train_evaluation(ml_est, X_test[features], y_test)

    yhat = np.where(X_test["yhat_llm"] > np.mean(y_train), 1, 0)
    print('accuracy llm: ', accuracy_score(y_test, yhat))
    print('f1_score llm: ', f1_score(y_test, yhat))
    print('mean(yhat_llm): ', np.mean(yhat))

    ml_est = training(df_train_full[features], y)
    train_evaluation(ml_est, df_train_full[features], y)

    yhat = np.where(df_train_full["yhat_llm"] > np.mean(y), 1, 0)
    print('accuracy llm: ', accuracy_score(y, yhat))
    print('f1_score llm: ', f1_score(y, yhat))
    print('mean(yhat_llm): ', np.mean(yhat))

    yhat = ml_est.predict(df_test[features])
    pd.concat([df_test["id"], pd.Series(yhat, name="target")], axis=1).to_csv("submission.csv", index=False)

    embed()


def main(args):
    np.random.seed(42)

    llm_run()

    # numerical_model()


if __name__ == "__main__":
    main(sys.argv[1:])
