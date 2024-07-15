import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.bpe import get_encoder

from IPython import embed


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


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            output = model.generate(input_ids.to(device)).cpu().detach().numpy().tolist()
            yhat += np.array(output).squeeze().tolist()

    df["yhat"] = yhat
    return df


def padding(input_ids, max_length):
    for i, ids in enumerate(input_ids):
        pad_list = [666] * (max_length - len(ids)) # negative values throw error in embeddings
        input_ids[i] = ids + pad_list
    return input_ids


def encode_text(df, enc):
    input_ids = []
    for text in df["text"].tolist():
        input_ids.append(enc.encode(text))
    return input_ids


def main(pretrained):
    np.random.seed(42)
    torch.manual_seed(666)

    df_train_full = pd.read_csv("../train.csv")

    df_train_full = df_train_full[["text", "target"]]

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    enc = get_encoder()
    input_ids_train = encode_text(df_train, enc)

    max_length = 0
    for ids in input_ids_train:
        len_ids = len(ids)
        if len_ids > max_length:
            max_length = len_ids

    input_ids_train = padding(input_ids_train, max_length)

    train_dataset = TensorDataset(
        torch.tensor(input_ids_train), 
        torch.tensor(df_train["target"].tolist(), dtype=torch.long)
        )

    # create a GPT instance
    if pretrained:
        model = GPT.from_pretrained('gpt2', 2)
    else:
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_length # 1024 is openai's model block_size
        model_config.n_output_nodes = 2
        model = GPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 10000
    # train_config.epochs = 10
    # 0.72 F1 score for training from scratch
    # (0.83 with encoder-LLM finetuning)
    train_config.epochs = 5
    # 0.76 for GPT2 finetuning (--pretrained)
    train_config.num_workers = 0
    train_config.batch_size = 32
    trainer = Trainer(train_config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    # validation
    df_train = predict(model, DataLoader(train_dataset, batch_size=32), df_train)
    evaluation(df_train["target"], df_train["yhat"])

    input_ids_val = encode_text(df_val, enc)
    input_ids_val = padding(input_ids_val, max_length)
    val_dataset = TensorDataset(
        torch.tensor(input_ids_val), 
        torch.tensor(df_val["target"].tolist(), dtype=torch.long)
        )
    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val["target"], df_val["yhat"])

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.pretrained)
