# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def text2token(sentence, config):
    token = config.tokenizer.tokenize(sentence)
    token = [CLS] + token
    seq_len = len(token)
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    pad_size = config.pad_size
    if len(token) < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
        token_ids += ([0] * (pad_size - len(token)))
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size

    return token_ids, seq_len, mask


def build_data_for_predict(sentences, config):
    contents = []
    for line in sentences:
        lin = line.strip()

        # TODO:这里会不会有问题
        if not lin:
            continue
        _token_ids, _seq_len, _mask = text2token(lin, config)
        contents.append((_token_ids, _seq_len, _mask))

    x = torch.LongTensor([_[0] for _ in contents]).to(config.device)
    seq_len = torch.LongTensor([_[1] for _ in contents]).to(config.device)
    mask = torch.LongTensor([_[2] for _ in contents]).to(config.device)
    return x, seq_len, mask


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token_ids, seq_len, mask = text2token(content, config)
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def print2log(msg, logfile):
    print(msg)
    if logfile is None:
        return
    with open(logfile, 'a') as f:
        print(msg, file=f, flush=True)
