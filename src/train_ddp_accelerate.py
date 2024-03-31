from accelerate import Accelerator
from accelerate import PartialState
from torch.utils.checkpoint import checkpoint

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
import wandb


def main():
    accelerator = Accelerator()
    #torch.checkpoint.use_reentrant=True
    torch.backends.cuda.matmul.allow_tf32 = True

    """# Define Hyperparameters"""
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
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 0
    logging_steps = 25
    max_seq_length = None
    packing = False


    wandb.init(
    project="llm_training",
    config={
        "model_name": model_name,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optim": optim,
        "weight_decay": weight_decay,
    }
    )





    ####################################################
    dataset = load_dataset(dataset_name, split="train")

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

    #device_string = PartialState().process_index

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #device_map={'':device_string}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        #gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        seed=42,
        ddp_find_unused_parameters=False,
    )



    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir+new_model)
    # Empty VRAM
    del model
    del trainer
    import gc
    gc.collect()
    wandb.finish()


if __name__ == "__main__":
    main()
