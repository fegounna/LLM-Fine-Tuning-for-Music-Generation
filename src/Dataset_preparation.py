import os
from tqdm import tqdm
from midi_text import midi_to_text
import pandas as pd

prompts =  []
responses = []

directory = "../midis"

for file in tqdm(os.listdir(directory)):
  if file.endswith(".midi") or file.endswith(".mid"):
    s = midi_to_text(directory+'/'+file)
    n = len(s)
    if(n>0):
      i = n//10
      while(s[i]!=' '):
        i +=1
      S1 = s[:i]
      S2 = s[i+1:]

      prompts.append(S1)
      responses.append(S2)

df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_json('../Dataset/train.jsonl', orient='records', lines=True)
test_df.to_json('../Dataset/test.jsonl', orient='records', lines=True)
