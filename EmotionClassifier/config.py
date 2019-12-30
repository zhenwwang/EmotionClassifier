"""
hyper parameters

@author: Zhenwei Wang
@date: 12/29/2019
"""
mark = 1
SEED = 1234
gpuid = 0

# data
batch_size = 8
vocab_size = 40000
data_path = 'data'
train_file = 'train.txt'
valid_file = 'valid.txt'
test_file = 'test.txt'
text_field = 'text_field.pkl'
label_field = 'label_field.pkl'
custom_word_embedding = 'sgns.weibo.bigram-char'

# model
bidirectional = True
embedding_dim = 300
hidden_dim = 256
n_layers = 2
dropout = 0.5
lr = 0.001  # learning rate
epochs = 50
