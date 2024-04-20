import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from peft import PeftConfig, PeftModel
from unsloth import FastLanguageModel

peft_model_id = '/users/eleves-a/2022/yessin.moakher/output/llama-2-music_4k/'
device_map = {"": 0}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = peft_model_id, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)


# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


#system_message ="""You are a classical pianist composer. In this context, each music note in a musical sequence is described using four parameters: pitch (p) from 0 to 127 (highest pitch), volume (v) from 0 to 127 (loudest), duration of the note (d) in ticks, and the length of the pause (t) in ticks before the next note begins regardless of the previous note's duration. A tick is approximately 5.21 milliseconds. Each parameter is followed by its value and separated by colons (e.g. p52:v57:d195:t212). Your composition should demonstrate a clear progression and development, appropriate pauses, including thoughtful variations in melody, harmony, rhythm.
#Your Task is to complete the generation of :"""

system_message ="""You are a classical pianist composer. In this context, each music note in a musical sequence is described using four parameters: pitch (p) from 0 to 127 (highest pitch), volume (v) from 0 to 127 (loudest), duration of the note (d) in ticks, and the length of the pause (t) in ticks before the next note begins regardless of the previous note's duration. A tick is approximately 5.21 milliseconds. Each parameter is followed by its value and separated by colons (e.g. p52:v57:d195:t212). Your composition should demonstrate a clear progression and development, appropriate pauses, including thoughtful variations in melody, harmony, rhythm. Your Task is to complete the generation of :"""
s=""
#prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????
prompt = f"[INST] <<SYS>> {system_message} <</SYS>> {s} [/INST]" 

#inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

#outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
#print(tokenizer.batch_decode(outputs))

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, temperature =0.8,repetition_penalty=0.5,max_length=1028)
                #penalty_alpha=0.6, max_length=1028)
                #
result = pipe(prompt)
print(result[0]['generated_text'])
