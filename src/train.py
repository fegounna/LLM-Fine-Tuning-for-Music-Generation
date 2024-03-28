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


#For DDP
from accelerate import PartialState
import torch.multiprocessing as mp
#from torch .utils.data.distributed  import DistributedSampler
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



def ddp_setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def main(rank: int, world_size: int):

    #Put this in Yaml File
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
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
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
    device_map = {"": 0}

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

    #model = DDP(model, device_ids=[device_map]) #For DDP

    #LORA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

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
        evaluation_strategy="steps",
        eval_steps=5,  # Evaluate every 20 steps
        report_to="wandb",
        seed=42,
        gradient_checkpointing_kwargs={'use_reentrant':False}, #For DDP
    )




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

    #ddp_setup(rank,world_size)

    device_string = PartialState().process_index #For DDP
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #device_map=device_map #without DDP
        device_map={'':device_string}, #For DDP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    #Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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
    destroy_process_group() #For DDP
    wandb.finish()


if __name__ == "__main__":
    world_size = 2
    local_rank = 0
    init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)
