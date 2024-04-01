import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from peft import PeftConfig, PeftModel



peft_model_id = '/users/eleves-a/2022/yessin.moakher/llama-2-7b-music-smidi/'
model_id = "NousResearch/llama-2-7b-chat-hf"
device_map = {"": 0}
#Get PeftConfig from the finetuned Peft Model. This config file contains the path to the base model

# If you quantized the model while finetuning using bits and bytes 
# and want to load the model in 4bit for inference use the following code.
# NOTE: Make sure the quant and compute types match what you did during finetuning

compute_dtype = getattr(torch, "float16")
#QLORA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

###
logging.set_verbosity_info()
#Load the base model - if you are not using the bnb_config then remove the quantization_config argument
#You may or may not need to set use_auth_token to True depending on your model.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    #use_auth_token=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = PeftModel.from_pretrained(model, peft_model_id)

# Ignore warnings
#logging.set_verbosity(logging.CRITICAL)


system_message ="""A piece of music is a set of music notes that are represented by quadruplets. Within each, the 4 variables(separated by ":") are p (pitch), d (duration),v (velocity) and t (time), followed by their value (example p52:v5:d1895:t212). 
p corresponds to the pitch of the note (example p60 for a C3). The pitch difference between 2 notes is the number of semitones that separate them / v the velocity note, the volume of the played note / d duration note, the duration (in milliseconds) of the note to be heard / t the time that separates the instant when the note is played from the instant when the next note will be played.
Your job is to complete the composition of """

s = """p60:v64:d500:t600 p62:v64:d250:t250 p64:v64:d250:t500 p65:v64:d1000:t700"""

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
result = pipe(prompt)
print(result[0]['generated_text'])
