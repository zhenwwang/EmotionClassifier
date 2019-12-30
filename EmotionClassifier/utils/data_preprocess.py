"""
raw_data.txt -> train/valid/text.txt

include:
    - remove some special tokens such as "（转）（图片）//XXX@: 回复@XXX:"
    - tokenize
    - keep the top K most frequent words

@author: Zhenwei Wang
@date: 12/29/2019
"""
# coding=utf-8
import random

from pyhanlp import HanLP
import re
import os


def clean_text(text):
    text = text.strip()
    # remove some special tokens such as "（转）（图片）//@XXX: 回复@XXX:"
    text = re.sub(r"（转）", r"", text)
    text = re.sub(r"（图）", r"", text)
    text = re.sub(r"//@.*(:|：)", r"", text)
    text = re.sub(r"回复@.*(:|：)", r"", text)

    # tokenize
    res = []
    for term in HanLP.segment(text):
        res.append(term.word)
    return res


def write_to_file(token_list, label, writer):
    for i in range(len(token_list)):
        writer.write(token_list[i])
        if i != len(token_list) - 1: writer.write(' ')
    writer.write('\t' + label)


if __name__ == '__main__':
    random.seed(1234)
    with open(os.path.join('..', 'data', 'raw_data.txt'), 'r') as input \
            , open(os.path.join('..', 'data', 'train.txt'), 'w') as train \
            , open(os.path.join('..', 'data', 'valid.txt'), 'w') as valid \
            , open(os.path.join('..', 'data', 'test.txt'), 'w') as test:
        data = input.readlines()
        train_size = int(len(data) * 0.8)
        valid_size = int(len(data) * 0.1)
        test_size = int(len(data) * 0.1)
        random.shuffle(data)
        i = 0
        for sentence in data:
            text, label = sentence.split('\t')
            text = clean_text(text)
            if i < valid_size:
                writer = valid
            elif i < valid_size + test_size:
                writer = test
            else:
                writer = train
            write_to_file(text, label, writer)
            i+=1
