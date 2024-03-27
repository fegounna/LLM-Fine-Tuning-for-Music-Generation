import os
from tqdm import tqdm
from midi_textefinal import texte
import pandas as pd
import random

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
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})
print("number of errors is ",error_counter)
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_json('./Dataset/train.jsonl', orient='records', lines=True)
test_df.to_json('./Dataset/test.jsonl', orient='records', lines=True)
