import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)


model_name = "fegounna/llama-2-7b-music-smidi"

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained("NousResearch/llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n???????[/INST]" # replace the ????
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
                #, max_length=200)
result = pipe(prompt)
print(result[0]['generated_text'])
