# coding: UTF-8
from pybert.models import bert
from pybert.train_eval import load_and_train

dataset = 'THUCNews'  # 数据集
logfile = 'log.txt'  # 日志文件
config = bert.Config(dataset, logfile=logfile)
load_and_train(config)
