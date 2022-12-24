# coding: UTF-8
import pybert.models.bert as bert
from pybert.train_eval import Prediction

config = bert.Config(dataset='THUCNews')

prediction = Prediction(config)

sentences = ['野兽用纪录打爆第二中锋 掘金版三巨头已巍然成型']  # * 128

predict_label, score = prediction.predict(sentences)
print("predict label:")
print(predict_label)
