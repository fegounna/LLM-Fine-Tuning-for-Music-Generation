import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from peft import PeftConfig, PeftModel
from unsloth import FastLanguageModel

peft_model_id = '/users/eleves-a/2022/yessin.moakher/output/llama-2-music_4k/'
device_map = {"": 0}

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-bnb-4bit",
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
#FastLanguageModel.for_inference(base_model)
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


system_message ="""A piece of music is a set of music notes that are represented by quadruplets. Within each, the 4 variables(separated by ":") are p (pitch), d (duration),v (velocity) and t (time), followed by their value (example p52:v5:d1895:t212). 
p corresponds to the pitch of the note (example p60 for a C3). The pitch difference between 2 notes is the number of semitones that separate them / v the velocity note, the volume of the played note / d duration note, the duration (in milliseconds) of the note to be heard / t the time that separates the instant when the note is played from the instant when the next note will be played.
Your job is to complete the composition of """


s = """p45:v73:d389:t1 p84:v84:d388:t7 p76:v62:d227:t0 p48:v63:d5:t110 p52:v69:d394:t5 p81:v78:d389:t117 p76:v68:d157:t5 p57:v60:d397:t154 p84:v87:d405:t1 p45:v79:d404:t1 p76:v67:d241:t121 p81:v79:d289:t1 p52:v60:d404:t123 p57:v69:d403:t3 p76:v73:d162:t159 p84:v90:d403:t1"""

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))

#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer,penalty_alpha=0.7, top_k=10, max_length=4096)
#result = pipe(prompt)
#print(result[0]['generated_text'])
