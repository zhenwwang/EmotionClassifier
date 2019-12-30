"""
test the model with lowest loss on valid

@author: Zhenwei Wang
@date: 12/30/2019
"""
from torch import nn

from config import *
import torch
import os

from utils.dataloader import get_data_iterator
from utils.evaluate import evaluate

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = torch.device('cuda', gpuid)

    # get data_iterator of train/valid/test
    print('start getting data_iterator of train/valid/test...')
    TEXT = torch.load(os.path.join(data_path, text_field))
    LABEL = torch.load(os.path.join(data_path, label_field))
    train_iterator, valid_iterator, test_iterator = get_data_iterator(device,
                                                                      TEXT,
                                                                      LABEL,
                                                                      batch_size,
                                                                      data_path,
                                                                      train_file,
                                                                      valid_file,
                                                                      test_file)
    print('=================== success ===================')

    print('start loading model...')
    model = torch.load(os.path.join('saved_models', f'model_{mark}.pkl'))
    criterion = nn.CrossEntropyLoss().to(device)
    print('=================== success ===================')

    print('start testing...')
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
