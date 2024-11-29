import os
import json
import random
import pandas as pd

questions = pd.read_table(os.path.expanduser("mmbench_dev_20230712.tsv"))
questions = questions.sample(n=300, random_state=0) # random seeds

val_questions = questions.iloc[:100]
test_questions = questions.iloc[100:]

print("Eval len: ", len(val_questions))
print("Test len: ", len(test_questions))

data = pd.DataFrame(val_questions)
data.to_csv('./val_mmbench.tsv', sep='\t', index=False, header=True)

data = pd.DataFrame(test_questions)
data.to_csv('./test_mmbench.tsv', sep='\t', index=False, header=True)
