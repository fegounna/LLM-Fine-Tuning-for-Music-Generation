import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.huggingface.transformers
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

dataset = load_dataset(dataset_name, split="train")
ray_train_ds = ray.data.from_huggingface(dataset) 

def train_func():
    model_name = "NousResearch/llama-2-7b-chat-hf"
    new_model = "llama-2-7b-music-smidi"
    dataset_name = "fegounna/GMP"
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    output_dir = "/Data/Models/"
    num_train_epochs = 1
    fp16 = False
    bf16 = False
    gradient_checkpointing = False
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 0
    logging_steps = 25
    max_seq_length = None
    packing = False
    device_map = {"": 0}


    ####################################################   

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    #QLORA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    #####################

    #LORA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map 
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    #Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = ray.train.get_dataset_shard("train")
    train_iterable_ds = train_dataset.iter_torch_batches(batch_size=4)

    


    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        seed=42,
    )



    trainer = SFTTrainer(
        model=model,
        train_dataset=train_iterable_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.add_callback(RayTrainReportCallback())

    trainer = prepare_trainer(trainer)

    trainer.train()



if __name__ == "__main__":
    # Init Ray cluster
    ray.init(address="auto")

    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        datasets={"train": ray_train_ds},
    )
    result = ray_trainer.fit()
