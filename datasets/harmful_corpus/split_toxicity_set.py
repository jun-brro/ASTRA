import json
import random
import pandas as pd

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

toxic_prompts = rtp_read("./rtp_prompts.jsonl")

random.seed(0) # random seeds
toxic_prompts = random.sample(toxic_prompts, 150)
val_toxic_prompts = toxic_prompts[:50]
test_toxic_prompts = toxic_prompts[50:]

print("Eval len: ", len(val_toxic_prompts))
print("Test len: ", len(test_toxic_prompts))

data = pd.DataFrame(val_toxic_prompts, columns=['prompts'])
data.to_csv('./val_rtp_prompts.csv', index=False)

data = pd.DataFrame(test_toxic_prompts, columns=['prompts'])
data.to_csv('./test_rtp_prompts.csv', index=False)