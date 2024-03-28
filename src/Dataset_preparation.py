import os
from tqdm import tqdm
from midi_textefinal import texte
import pandas as pd
import random
from datasets import  DatasetDict, Dataset

prompts =  []
responses = []

directory = "./midis"
error_counter = 0

for file in tqdm(os.listdir(directory)):
  if file.endswith(".midi") or file.endswith(".mid"):
    try:
      s = texte(directory+'/'+file)
      n = len(s)
      if(n>500):
        curr = min(3000,n)
        while(curr < n and s[curr]!=" "):
          curr += 1
        s = s[:curr]
        i = curr // 10
        while(s[i]!=' '):
          i +=1
        S1 = s[:i]
        S2 = s[i+1:]
  
        prompts.append(S1)
        responses.append(S2)
    except Exception as e:
      print(e)
      error_counter += 1
      continue
data = {
    'prompt': prompts,
    'response': responses
}
print("number of errors is ",error_counter)

dataset = Dataset.from_dict(data)

system_message ="""A piece of music is a set of music notes that are represented by quadruplets. Within each, the 4 variables(separated by ":") are p (pitch), d (duration),v (velocity) and t (time), followed by their value (example p52:v5:d1895:t212). 
p corresponds to the pitch of the note (example p60 for a C3). The pitch difference between 2 notes is the number of semitones that separate them / v the velocity note, the volume of the played note / d duration note, the duration (in milliseconds) of the note to be heard / t the time that separates the instant when the note is played from the instant when the next note will be played.
Your job is to complete the composition of """

dataset_mapped = dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

dataset_mapped.push_to_hub("fegounna/GMP")




#train_df = df.sample(frac=0.9, random_state=42)
#test_df = df.drop(train_df.index)

#train_df.to_json('./Dataset/train.jsonl', orient='records', lines=True)
#test_df.to_json('./Dataset/test.jsonl', orient='records', lines=True)
