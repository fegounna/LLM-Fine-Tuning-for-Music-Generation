import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import wandb
from unsloth import FastLanguageModel

def main():
    new_model = "llama-2-GMP-8k_mlp_e"
    max_seq_length = 8192 
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True 
    dataset_name = "fegounna/GMP_8K"
    output_dir = "/users/eleves-a/2022/yessin.moakher/output/"
    num_train_epochs = 6
    max_steps = -1
    fp16 = not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_bf16_supported()
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 1 
    optim = "paged_adamw_32bit" #last
    save_steps = -1
    save_strategy="epoch"
    learning_rate = 2e-4
    max_grad_norm = 0.3
    group_by_length = True
    warmup_ratio = 0.1
    #warmup_ratio = 0.01
    #warmup_steps = 50 
    #set the final learning rate to be 1/30th 
    lr_scheduler_type = "cosine_with_restarts"
    weight_decay = 0.001 #make it 0
    #weight_decay = 0
    logging_steps = 5
    packing = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        use_rslora = False,  
        loftq_config = None, 
    )

    dataset = load_dataset(dataset_name, split="train[:45000]")
    
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
        #warmup_steps=warmup_steps,
        save_strategy=save_strategy,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        seed=42,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.train(resume_from_checkpoint = True)
    trainer.model.save_pretrained(output_dir+new_model)
    # Empty VRAM
    del model
    del trainer
    import gc
    gc.collect()
if __name__ == "__main__":
    main()
