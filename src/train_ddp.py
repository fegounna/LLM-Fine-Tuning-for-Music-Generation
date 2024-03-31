import ray
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig
import ray.train.huggingface.transformers
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)
from torch.utils.checkpoint import checkpoint


import os
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    default_data_collator,
)
from peft import LoraConfig, PeftModel, get_peft_model

model_name = "NousResearch/llama-2-7b-chat-hf"
dataset_name = "fegounna/GMP"
dataset = load_dataset(dataset_name, split="train")
ray_train_ds = ray.data.from_huggingface(dataset)

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ret =  tokenizer(list(examples["text"]), padding="max_length", truncation=True, max_length=1024, return_tensors="np")
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)

tokenized_ray_dataset = ray_train_ds.map_batches(tokenize_function)
tokenized_ray_dataset  = {"train": tokenized_ray_dataset}


def train_func():
    torch.checkpoint.use_reentrant=False
    torch.backends.cuda.matmul.allow_tf32 = True
    model_name = "NousResearch/llama-2-7b-chat-hf"
    new_model = "llama-2-7b-music-smidi"
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
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = 1000
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
    #peft
    
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)

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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        seed=42,
        report_to="none",
        label_names=["input_ids", "attention_mask"],
        push_to_hub=False,
        ddp_find_unused_parameters=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    trainer = Trainer(
        model=model,
        train_dataset=train_iterable_ds,
        #peft_config=peft_config,
        #dataset_text_field="text",
        #max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        #packing=packing,
        data_collator=default_data_collator,
    )
    trainer.add_callback(RayTrainReportCallback())

    trainer = prepare_trainer(trainer)

    trainer.train()



if __name__ == "__main__":
    # Init Ray cluster
    ray.init(address="auto", ignore_reinit_error=True)
    storage_path = "/users/eleves-a/2022/yessin.moakher/Models"
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        datasets=tokenized_ray_dataset,
        run_config=RunConfig(storage_path=storage_path),
    )
    result = ray_trainer.fit()
