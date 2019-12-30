"""
predict

@author: Zhenwei Wang
@date: 12/30/2019
"""
import os
import torch
from config import *

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
def predict_class(model, tokens, TEXT, device, min_len=4):
    model.eval()
    if len(tokens) < min_len:
        tokens += ['<pad>'] * (min_len - len(tokens))
    indexed = [TEXT.vocab.stoi[t] for t in tokens]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)
    return max_preds.item()


if __name__ == '__main__':
    device = torch.device('cuda', gpuid)
    TEXT = torch.load(os.path.join(data_path, text_field))
    LABEL = torch.load(os.path.join(data_path, label_field))
    model = torch.load(os.path.join('saved_models', f'model_{mark}.pkl'))
    pred_class = predict_class(model, "我 生气 生气 了 ！", TEXT, device)
    print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')
    print(pred_class)
