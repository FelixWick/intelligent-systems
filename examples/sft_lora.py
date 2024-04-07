import os

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset


access_token = os.environ['HF_TOKEN']


def finetuning():
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)

    data = load_dataset("mehdiiraqui/twitter_disaster")
    data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

    def formatting_func(example):
        text = f"Tweet: {example['text'][0]}\nDisaster: {example['target'][0]}"
        return [text]

    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.1,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            output_dir=".",
            num_train_epochs=1,
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    model.save_pretrained(".")

    input_text = "Tweet: Heavy accident on Interstate 10.\nDisaster: "
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=1)
    print(tokenizer.decode(outputs[0]))

    input_text = "Tweet: Good morning everybody.\nDisaster: "
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=1)
    print(tokenizer.decode(outputs[0]))

    # quiz question ;): Why is it not giving the correct answer?


if __name__ == "__main__":
    finetuning()