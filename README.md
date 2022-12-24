# pybert
## 预训练语言模型
bert模型放在 bert_pretain目录下，三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   
备用：模型的网盘地址：https://pan.baidu.com/s/123VJOsbHyi6RW0zyvOG0Ow?pwd=lq7a

## 训练数据下载

[THUCNews](https://github.com/guofei9987/datasets_for_ml/blob/master/nlp/THUCNews.7z)

## 训练和预测

训练
```python
from pybert.models import bert
from pybert.train_eval import load_and_train

dataset = 'THUCNews'  # 数据集
logfile = 'log.txt'  # 日志文件
config = bert.Config(dataset, logfile=logfile)
load_and_train(config)
```

预测
```python
# coding: UTF-8
import pybert.models.bert as bert
from pybert.train_eval import Prediction

config = bert.Config(dataset='THUCNews')
prediction = Prediction(config)

sentences = ['野兽用纪录打爆第二中锋 掘金版三巨头已巍然成型']  # * 128

predict_label, score = prediction.predict(sentences)
print("predict label:")
print(predict_label)
```




## 对应论文
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
