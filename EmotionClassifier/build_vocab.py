"""
use pretrained word embedding and train data to train text field and
label field and saved

@author: Zhenwei Wang
@date: 12/30/2019
"""
import os
import torch
from torchtext import data
from config import *
from utils.load_custom_embeddings import load_custom_embeddings

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def build_vocab(vocab_size,
                path='data',
                train='train.txt'):
    # include_lengths=True
    # batch.text will be a tuple with the first element being our
    # sentence (a numericalized tensor that has been padded)
    # and the second element being the actual lengths of our sentences.
    TEXT = data.Field(include_lengths=True)
    LABEL = data.LabelField()  # LongTensors when multi-class
    """
    fields definition:
        - the first element of these inner tuples will become the batch object's attribute name
        - the second element is the Field name
    """
    fields = [('text', TEXT), ('label', LABEL)]

    # create dataset
    train_data = data.TabularDataset.splits(
        path=path,
        train=train,
        format='tsv',
        fields=fields
    )
    train_data = train_data[0]  # 返回的是tuple(train_data)

    # build vocab with custom word vector
    custom_embeddings = load_custom_embeddings()
    TEXT.build_vocab(train_data,
                     max_size=vocab_size,
                     vectors=custom_embeddings
                     )
    LABEL.build_vocab(train_data)

    print(f'vocab size: {len(TEXT.vocab)}')

    print('saving....')
    torch.save(TEXT, os.path.join(data_path, text_field))
    torch.save(LABEL, os.path.join(data_path, label_field))
    print('success saved')


if __name__ == '__main__':
    build_vocab(vocab_size, data_path, train_file)
