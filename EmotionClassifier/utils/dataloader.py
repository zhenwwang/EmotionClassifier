"""
data.txt -> data iterator

@author: Zhenwei Wang
@date: 12/29/2019
"""
import torch
from torchtext import data
from torchtext import datasets


def get_data_iterator(device,
                      TEXT,
                      LABEL,
                      batch_size,
                      path='data',
                      train='train.txt',
                      validation='valid.txt',
                      test='test.txt'):
    fields = [('text', TEXT), ('label', LABEL)]

    # create dataset
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path=path,
        train=train,
        validation=validation,
        test=test,
        format='tsv',
        fields=fields
    )

    # create iterator
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_within_batch=True,
        batch_size=batch_size,
        device=device)

    return train_iterator, valid_iterator, test_iterator
