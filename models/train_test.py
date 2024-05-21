import torch


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    edge_index = data[3]
    model.train()
    for epoch in range(1000):
        out = model(data, edge_index)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def test(model, data):
    model.eval()
    adj = data[3]
    for epoch in range(1000):
        _, pred = model(data, adj).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
