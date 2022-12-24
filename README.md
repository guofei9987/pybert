# pybert

安装
```bash
>pip install pybert
```


## 预训练模型


下载地址：  
- bert_Chinese 模型文件: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
- 词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
- 【备用】百度网盘：https://pan.baidu.com/s/1HPZBvkMAyu0nDUqHWsb0SA?pwd=abbn

所需文件：
- pytorch_model.bin  
- bert_config.json  
- vocab.txt  


放到 bert_pretrain 文件夹中

## 训练数据下载

[THUCNews](https://github.com/guofei9987/datasets_for_ml/blob/master/nlp/THUCNews.7z)
- 可以任意指定文件夹名称，训练数据的格式要和上面一致


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

sentences = ['野兽用纪录打爆第二中锋 掘金版三巨头已巍然成型', '56所高校预估2009年湖北录取分数线出炉']

predict_label, score = prediction.predict(sentences)
print("predict label:")
print(predict_label)
```




## 对应论文
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
