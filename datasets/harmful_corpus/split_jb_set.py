import os
import json
import random
import pandas as pd

questions = pd.read_csv("eval.csv")
# questions = questions.sample(n=300, random_state=0) # random seeds

test_questions = questions.iloc[:len(questions)//2]
val_questions = questions.iloc[len(questions)//2:]

print("Eval len: ", len(test_questions))
print("Test len: ", len(val_questions))

data = pd.DataFrame(test_questions)
data.to_csv('./test.csv', index=False, header=True)

data = pd.DataFrame(val_questions)
data.to_csv('./val.csv', index=False, header=True)
