# coding: UTF-8
import pybert.models.bert as bert
from pybert.train_eval import Prediction
import random

config = bert.Config(dataset='THUCNews')

prediction = Prediction(config)

with open('THUCNews/data/test.txt') as f:
    contents = f.readlines()

contents = [line.strip().split('\t') for line in contents]

# %%
data_pick = random.choices(contents, k=100)

sentences = [_[0] for _ in data_pick]
true_label = [int(_[1]) for _ in data_pick]

predict_label, _ = prediction.predict(sentences)

predict_label = predict_label.tolist()

print(predict_label)
print(true_label)

diff = [predict_label[i] == true_label[i] for i in range(len(predict_label))]
print(diff)

acc = sum(diff) / len(predict_label)
print(acc)
assert acc > 0.9
