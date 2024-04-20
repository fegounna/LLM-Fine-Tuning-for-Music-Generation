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


system_message ="""You are a classical pianist composer. In this context, each music note in a musical sequence is described using four parameters: pitch (p) from 0 to 127 (highest pitch), volume (v) from 0 to 127 (loudest), duration of the note (d) in ticks, and the length of the pause (t) in ticks before the next note begins regardless of the previous note's duration. A tick is approximately 5.21 milliseconds. Each parameter is followed by its value and separated by colons (e.g. p52:v57:d195:t212). Your composition should demonstrate a clear progression and development, appropriate pauses, including thoughtful variations in melody, harmony, rhythm.
Your Task is to complete the generation of :"""


s = """p45:v73:d389:t1 p84:v84:d388:t7 p76:v62:d227:t0 p48:v63:d5:t110 p52:v69:d394:t5 p81:v78:d389:t117 p76:v68:d157:t5 p57:v60:d397:t154 p84:v87:d405:t1 p45:v79:d404:t1 p76:v67:d241:t121 p81:v79:d289:t1 p52:v60:d404:t123 p57:v69:d403:t3 p76:v73:d162:t159 p84:v90:d403:t1"""

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
print(tokenizer.batch_decode(outputs))

#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer,penalty_alpha=0.7, top_k=10, max_length=4096)
#result = pipe(prompt)
#print(result[0]['generated_text'])
