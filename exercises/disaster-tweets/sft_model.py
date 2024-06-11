import os

from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import SFTTrainer, SFTConfig
from datasets import Dataset
# from peft import LoraConfig

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from IPython import embed


access_token = os.environ['HF_TOKEN']


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def finetuning():
    df_train_full = pd.read_csv("../train.csv")
    df_train_full = df_train_full[["text", "target"]]
    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    df_train["text"] = "<|user|> Is this news about a disaster? " + df_train["text"] + "</s> <|assistant|> " + df_train["target"].astype(str) + "</s>"
    df_val["text"] = "<|user|> Is this news about a disaster? " + df_val["text"] + "</s> <|assistant|> "

    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)

    model_id = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)

    sft_config = SFTConfig(
        output_dir=".",
        dataset_text_field="text",
        max_seq_length=350,
        num_train_epochs=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        args=sft_config,
        # peft_config=LoraConfig(),
    )
    trainer.train()

    yhat = []
    for test_prompt in dataset_val["text"]:
        input_ids = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(**input_ids, max_new_tokens=1)
        output_text = tokenizer.decode(outputs[0][-1])
        output_text = output_text.strip(' ')
        # print(output_text)
        if output_text not in ['0', '1']:
            output_text = '0'
            print("blabla")
        output_text = int(output_text)
        yhat.append(output_text)

    evaluation(np.array(dataset_val["target"], dtype=int), np.array(yhat))

    embed()


if __name__ == "__main__":
    finetuning()
