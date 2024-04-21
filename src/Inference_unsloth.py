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
s="p48:v66:d272:t1 p84:v80:d272:t0 p76:v63:d272:t116 p52:v69:d156:t1 p81:v76:d155:t121 p45:v70:d42:t1 p84:v82:d41:t1 p76:v63:d41:t114 p81:v79:d40:t0 p52:v69:d40:t1 p48:v66:d39:t1 p76:v61:d39:t117 p84:v81:d37:t1 p52:v68:d38:t1 p48:v65:d37:t0 p76:v62:d37:t115 p81:v77:d39:t0 p52:v67:d38:t2 p48:v63:d37:t1 p76:v61:d37:t116 p84:v80:d36:t1 p52:v67:d36:t1 p48:v63:d35:t1 p76:v60:d35:t117 p81:v76:d37:t0 p52:v66:d37:t1 p48:v63:d37:t1 p76:v60:d37:t117 p84:v80:d36:t1 p52:v67:d35:t0 p48:v63:d35:t1 p76:v61:d35:t117 p81:v77:d36:t0 p52:v67:d35:t0 p48:v63:d35:t1 p76:v60:d35:t117 p84:v81:d35:t1 p52:v68:d34:t1 p48:v63:d34:t0 p76:v60:d34:t118 p81:v76:d35:t0 p52:v67:d35:t1 p48:v63:d35:t1 p76:v60:d34:t117 p84:v80:d34:t0 p52:v66:d34:t1 p48:v62:d33:t0 p76:v61:d33:t118 p81:v76:d34:t0 p52:v66:d34:t1 p48:v63:d34:t1 p76:v60:d34:t117 p84:v81:d34:t0 p52:v67:d34:t1 p48:v63:d33:t1 p76:v61:d33:t117 p81:v76:d34:t0 p52:v67:d33:t1 p48:v63:d33:t1 p76:v61:d33:t117 p84:v81:d33:t1 p52:v67:d33:t0 p48:v63:d32:t1 p76:v60:d32:t118 p81:v76:d33:t1 p52:v66:d32:t0 p48:v62:d32:t1 p76:v60:d32:t118 p84:v81:d32:t1 p52:v66:d32:t1 p48:v63:d31:t1 p76:v61:d31:t118 p81:v76:d33:t0 p52:v66:d33:t1 p48:v62:d33:t1 p76:v60:d32:t118 p84:v81:d34:t1 p52:v66:d33:t1 p48:v62:d32:t0 p76:v60:d32:t118 p81:v76:d33:t0 p52:v66:d33:t1 p48:v62:d32:t1 p76:v60:d32:t117 p84:v81:d32:t1 p52:v66:d31:t1 p48:v62:d30:t0 p76:v60:d30:t118 p81:v76:d32:t1 p52:v66:d31:t0 p48:v62:d31:t1 p76:v60:d31:t118 p84:v81:d31:t1 p52:v66:d30:t1 p48:v62:d29:t1 p76:v60:d29:t117 p81:v76:d31:t1 p52:v66:d30:t1 p48:v62:d29:t1 p76:v60:d29:t117 p84:v81:d31:t1 p52:v66:d30:t0 p48:v62:d29:t1 p76:v60:d29:t117 p81:v76:d31:t1 p52:v66:d30:t1 p48:v62:d29:t1 p76:v60:d29:t117 p84:v81:d30:t0 p52:v66:d30:t1 p48:v62:d29:t0 p76:v60:d29:t117 p81:v76:d31:t0 p52:v66:d30:t1 p48:v62:d29:t1 p76:v60:d29:t117 p84:v81:d30:t1 p52:v66:d29:t1 p48:v62:d29:t0 p76:v60:d29:t117"
#prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????
prompt = f"[INST] <<SYS>> {system_message} <</SYS>> {s} [/INST]" 

#inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

#outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
#print(tokenizer.batch_decode(outputs))
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, temperature =0.6 ,top_k=5,max_length=2048)

#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, temperature =0.9,penalty_alpha=0.6,top_p=0.5,max_length=1024)
                #,repetition_penalty=1.5
                # max_length=1028)
                #
result = pipe(prompt)
print(result[0]['generated_text'])
