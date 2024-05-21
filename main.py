import time
import torch
from utils.loader import load_data, load_model
from models.train_test import train, test


def main():
    datanames = ['Cora', 'CiteSeer', 'PubMed', 'PPI']
    modelnames = ['GCN', 'GAT', 'GraphSAGE']
    device = torch.device('cuda')
    print("dataset:", datanames[0])
    datas = load_data(datanames[0], device)

    model = load_model(modelnames[0], datas, device)
    start = time.time()
    train(model, datas, device)
    a = time.time() - start

    start = time.time()
    test(model, datas)
    b = time.time() - start
    print("train time,inference time:%.3f/%.3f" % (a, b))
if __name__ == '__main__':
    main()
