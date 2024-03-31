import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel



#Set Path to folder that contains adapter_config.json and the associated .bin files for the Peft model
peft_model_id = '/Data/Modelsllama-2-7b-music-smidi'
model_id = "NousResearch/llama-2-7b-chat-hf"
device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(peft_model_id, use_temp_dir=False)
tokenizer.push_to_hub(peft_model_id, use_temp_dir=False)