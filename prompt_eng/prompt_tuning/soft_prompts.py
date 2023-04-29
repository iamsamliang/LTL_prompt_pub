import pickle
import torch
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
import numpy as np
import json
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def preprocess_data(example):
    instruction = example["instruction"]
    ltl = example["ltl"]

    tokenized_source = tokenizer(instruction, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    tokenized_target = tokenizer(ltl, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

    return {
        "tok_instruction": tokenized_source["input_ids"],
        "attention_mask": tokenized_source["attention_mask"],
        "tok_ltl": tokenized_target["input_ids"],
    }

accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


class T5PromptTuning(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, prompt_length: int):
        super().__init__(config)
        self.prompt_length = prompt_length
        self.prompt_embeddings = torch.nn.Embedding(prompt_length, config.d_model)

        # freeze the model parameters÷
        for param in self.parameters():
            param.requires_grad = False

        # only train the prompt_embeddings
        for param in self.prompt_embeddings.parameters():
            param.requires_grad = True

    def forward(self, tok_instruction, attention_mask=None, tok_ltl=None):
        # Retrieve the prompt embeddings and broadcast to batch size
        batch_size = tok_instruction.shape[0]
        prompt_embeds = self.prompt_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # Generate the input embeddings
        input_embeddings = self.get_input_embeddings()(tok_instruction)

        # Prepend the prompt embeddings to the input_embeddings
        # inputs_embeds = torch.cat((prompts[:, :self.prompt_length], input_embeddings), dim=1)
        prompt_inputs_embedded = torch.cat((prompt_embeds, input_embeddings), dim=1)

        # need to modify the attention mask to also attend to the prompt_embeddings
        prompt_mask = torch.ones(batch_size, self.prompt_length, device=attention_mask.device)
        attention_mask = torch.cat((prompt_mask, attention_mask), dim=1)

        # (confirmed) self.config.pad_token_id == tokenizer.pad_token_id 
        # create the decoder_inputs_embeds
        decoder_input_ids = shift_tokens_right(tok_ltl, self.config.pad_token_id, self.config.decoder_start_token_id)
        # decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)


        # Call the original model's forward method with the modified inputs
        return super().forward(inputs_embeds=prompt_inputs_embedded, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

# Set up a custom Trainer for T5PromptTuning
class T5PromptTuningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Retrieve the tokenized instructions and LTL formulas (targets)
        tok_instruction = inputs["tok_instruction"]
        labels = inputs["tok_ltl"]
        attention_mask = inputs["attention_mask"]

        # Call the model's forward method with the input embeddings
        outputs = model(tok_instruction=tok_instruction, attention_mask=attention_mask, tok_ltl=labels)

        logits = outputs.get("logits")

        # calculate the loss using CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Batch the input tensors without padding
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([example[key] for example in features])
        return batch


if __name__ == '__main__':
    # print_gpu_utilization()

    # Load and preprocess your dataset
    data_files = {}
    data_files["train"] = "csv_data/train_utt_0.3_123.csv"
    data_files["test"] = "csv_data/test_utt_0.3_123.csv"
    extension = "csv"

    datasets = load_dataset(extension, data_files=data_files)


    model_name = "google/flan-t5-base" # or "google/flan-t5-xxl"

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    # Apply the preprocess_data function to tokenize the dataset
    train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=["instruction", "ltl", "pattern_type", "props"])
    test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=["instruction", "ltl", "pattern_type", "props"])

    train_dataset.set_format(type="torch", columns=["tok_instruction", "attention_mask", "tok_ltl"])
    test_dataset.set_format(type="torch", columns=["tok_instruction", "attention_mask", "tok_ltl"])

    # Load the pre-trained T5 configuration
    config = T5Config.from_pretrained(model_name)

    # Initialize the T5PromptTuning model with the pre-trained configuration
    # increasing beyond 20 only causes marginal gains according to paper
    # model = T5PromptTuning.from_pretrained(model_name, config=config, prompt_length=20)

    #### Load the model for evaluation
    model_path = "./ptuning_weights/checkpoint-49000"
    model = T5PromptTuning.from_pretrained(model_path, prompt_length=20)

    custom_collator = CustomDataCollator(tokenizer=tokenizer)

    mps_avail = False
    if torch.backends.mps.is_available():
        print("M1 GPUs available\n")
        mps_avail = True

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./ptuning_weights",
        num_train_epochs=20,
        per_device_train_batch_size=8,
        # gradient_accumulation_steps=4, #  uses less memory, slows training as steps increase
        gradient_checkpointing=True, # uses less memory but slows training by 20%
        # fp16=True, # speed up training but uses more memory. Only available on CUDA devices
        # bf16=True, # speed up training but uses more memory. Only available on Ampere devices
        logging_dir="./ptuning_logs",
        logging_steps=250,
        learning_rate=0.3,
        weight_decay=1e-5,
        optim="adafactor", # adamw_hf or adafactor
    #     optim_args={
    #     "learning_rate": 0.3,
    #     "weight_decay": 1e-5,
    #     # "beta2_decay": 0.8,
    #     "scale_parameter": False,
    # },
        optim_args="learning_rate=0.3, weight_decay=1e-5, scale_parameter=False",
        save_steps=500,
        save_strategy="steps",
        save_total_limit=5,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # eval_accumulation_steps=4,
        use_mps_device=mps_avail,
    )


    # Initialize the T5PromptTuningTrainer
    trainer = T5PromptTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collator,
    )

    # what device it is using
    device = trainer.model.device
    print(f"Model is on device: {device}")

    # Train the model
    # checkpoint_file = "./ptuning_weights/checkpoint-22000"
    # result = trainer.train(resume_from_checkpoint=checkpoint_file)
    # trainer.save_model("./ptuning_weights")

    eval_results = trainer.evaluate()

    with open("ptuning_result/evaluation_results.json", "w") as f:
        json.dump(eval_results, f)

    # print_summary(result)
