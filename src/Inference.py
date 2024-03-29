import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from peft import PeftConfig, PeftModel

#Set Path to folder that contains adapter_config.json and the associated .bin files for the Peft model
peft_model_id = '/Data/Models/llama-2-7b-music-smidi/'
model_id = "NousResearch/llama-2-7b-chat-hf"

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

#Load the base model - if you are not using the bnb_config then remove the quantization_config argument
#You may or may not need to set use_auth_token to True depending on your model.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    #use_auth_token=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the Peft/Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


system_message ="""A piece of music is a set of music notes that are represented by quadruplets. Within each, the 4 variables(separated by ":") are p (pitch), d (duration),v (velocity) and t (time), followed by their value (example p52:v5:d1895:t212). 
p corresponds to the pitch of the note (example p60 for a C3). The pitch difference between 2 notes is the number of semitones that separate them / v the velocity note, the volume of the played note / d duration note, the duration (in milliseconds) of the note to be heard / t the time that separates the instant when the note is played from the instant when the next note will be played.
Your job is to complete the composition of """

s = """p59:v72:d427:t1 p47:v68:d165:t8 p81:v65:d150:t1 p77:v74:d417:t159 p81:v71:d128:t2 p86:v83:d410:t5 p47:v75:d413:t1 p57:v68:d1333:t118 p53:v75:d424:t5 p83:v85:d412:t3 p81:v74:d723:t130 p59:v62:d425:t2 p77:v76:d154:t157 p86:v86:d427:t2 p77:v68:d264:t5 p47:v76:d435:t123 p83:v78:d420:t6 p53:v76:d421:t138 p77:v76:d422:t2 p59:v63:d404:t161 p86:v85:d458:t9 p81:v76:d111:t6 p47:v79:d241:t109 p83:v79:d933:t6 p53:v71:d743:t1 p81:v71:d350:t118 p59:v71:d214:t21 p77:v76:d227:t146 p45:v75:d411:t7 p84:v79:d150:t116 p81:v82:d411:t14 p52:v62:d397:t120 p57:v70:d407:t10 p76:v68:d397:t148 p45:v72:d426:t2 p84:v91:d424:t122 p52:v71:d302:t0 p93:v76:d10:t1 p81:v82:d432:t129 p57:v68:d425:t1 p76:v73:d424:t175 p84:v86:d418:t2 p45:v74:d416:t0 p52:v68:d547:t3 p96:v87:d6:t125 p81:v83:d358:t1 p93:v76:d11:t123 p57:v66:d410:t1 p76:v80:d210:t169 p84:v91:d440:t2 p96:v88:d8:t4 p45:v72:d1071:t121 p93:v82:d6:t2 p81:v84:d342:t3 p64:v68:d8:t1 p52:v72:d1467:t143 p57:v71:d509:t2 p69:v73:d146:t3 p76:v82:d251:t255 p76:v70:d242:t136 p71:v66:d405:t123 p67:v72:d413:t1 p57:v38:d166:t170 p76:v81:d388:t1 p57:v40:d241:t115 p71:v66:d341:t131 p67:v73:d294:t3 p57:v42:d145:t148 p95:v71:d5:t1 p76:v83:d219:t3 p57:v44:d631:t7 p88:v65:d9:t112 p71:v64:d24:t118 p67:v74:d148:t4 p52:v44:d175:t3 p79:v65:d218:t147 p76:v84:d163:t1 p59:v56:d247:t2 p67:v57:d237:t1 p88:v72:d52:t108 p71:v73:d105:t138 p67:v76:d1449:t0 p79:v63:d366:t1 p57:v54:d29:t1 p59:v58:d287:t191 p58:v53:d4:t1 p84:v90:d203:t0 p57:v55:d280:t2 p96:v86:d9:t107 p93:v76:d9:t1 p81:v83:d316:t145 p76:v70:d355:t13 p58:v33:d4:t10 p64:v25:d9:t6 p57:v27:d587:t139 p84:v89:d310:t4 p96:v85:d6:t1 p58:v57:d97:t102 p93:v75:d11:t1 p81:v81:d325:t150 p76:v89:d367:t1 p60:v51:d627:t1 p58:v53:d11:t2 p64:v44:d317:t5 p88:v71:d4:t145 p84:v91:d382:t2 p66:v59:d134:t0 p96:v88:d11:t0 p58:v56:d88:t108 p93:v79:d11:t1 p81:v83:d278:t132 p76:v82:d200:t5 p64:v33:d280:t2 p58:v46:d9:t7 p88:v57:d9:t131 p66:v57:d186:t0 p58:v59:d94:t2 p84:v89:d476:t1 p96:v86:d6:t1 p61:v52:d5:t125 p93:v82:d11:t0 p81:v90:d218:t4 p57:v31:d7:t129 p66:v45:d93:t2"""

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{s}[INST]" # replace the ????
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=3000)
result = pipe(prompt)
print(result[0]['generated_text'])
