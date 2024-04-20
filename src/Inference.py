import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from peft import PeftConfig, PeftModel



peft_model_id = '/users/eleves-a/2022/yessin.moakher/output/llama-2-music_4k/'
model_id = "/Data/Llama-2-7b-hf/"
device_map = {"": 0}
#Get PeftConfig from the finetuned Peft Model. This config file contains the path to the base model

# If you quantized the model while finetuning using bits and bytes 
# and want to load the model in 4bit for inference use the following code.
# NOTE: Make sure the quant and compute types match what you did during finetuning
"""
compute_dtype = getattr(torch, "float16")
#QLORA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)"""

###
#Load the base model - if you are not using the bnb_config then remove the quantization_config argument
"""model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    #use_auth_token=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)"""
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


system_message ="""A piece of music is a set of music notes that are represented by quadruplets. Within each, the 4 variables(separated by ":") are p (pitch), d (duration),v (velocity) and t (time), followed by their value (example p52:v5:d1895:t212). 
p corresponds to the pitch of the note (example p60 for a C3). The pitch difference between 2 notes is the number of semitones that separate them / v the velocity note, the volume of the played note / d duration note, the duration (in milliseconds) of the note to be heard / t the time that separates the instant when the note is played from the instant when the next note will be played.
Your job is to complete the composition of """


s = """p45:v73:d389:t1 p84:v84:d388:t7 p76:v62:d227:t0 p48:v63:d5:t110 p52:v69:d394:t5 p81:v78:d389:t117 p76:v68:d157:t5 p57:v60:d397:t154 p84:v87:d405:t1 p45:v79:d404:t1 p76:v67:d241:t121 p81:v79:d289:t1 p52:v60:d404:t123 p57:v69:d403:t3 p76:v73:d162:t159 p84:v90:d403:t1"""

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer,penalty_alpha=0.9, top_p = 0.6, max_length=4096)
result = pipe(prompt)
print(result[0]['generated_text'])
