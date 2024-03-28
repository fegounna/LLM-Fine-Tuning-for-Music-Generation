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
from accelerate import Accelerator

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    #ddp_find_unused_parameters=False
    # Hyperparameters (Consider moving hyperparameters to a config file)
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
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 25
    logging_steps = 5
    max_seq_length = None
    packing = False

    # Setup WandB (optional)
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

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    #QLORA config
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    # Load tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map 
        #device_map={'':device_string}, #For DDP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    #LORA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )


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
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=gradient_checkpointing,
        #evaluation_strategy="steps",
        #eval_steps=5,  # Evaluate every 20 steps
        report_to="wandb",
        seed=42,
        #gradient_checkpointing_kwargs={'use_reentrant':False}, #For DDP
    )

    # Initialize SFTTrainer
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

    # Prepare everything with accelerator
    model, trainer = accelerator.prepare(model, trainer)

    # Train
    trainer.train()

    # Save model (only from the main process)
    if accelerator.is_main_process:
        trainer.model.save_pretrained(output_dir + new_model)

    # Clean up
    wandb.finish()

if __name__ == "__main__":
    main()