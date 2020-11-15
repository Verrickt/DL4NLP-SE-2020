from models import GNNStack
import utils
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch
def loadData(name,args):
    dataset = Planetoid(root='/tmp/{}'.format(name), name=name)
    dataset = dataset.shuffle()
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataset,loader,test_loader
def train_model(model, loader, args):

    # build model
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    # train
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
       # print(total_loss)
    return model
def test_model(model, loader, name):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y
        mask = data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]
        correct += pred.eq(label).sum().item()
    
    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()
    acc = correct / total
    print('{}  test   '.format(name),acc)

    return correct / total
def score(args):
    time = 5
    meanAcc = 0
    names = ["Cora", "CITESEER"]

    for name in names:
        for i in range(time):
            dataset,train,test = loadData(name,args)
            model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task='node')
            model = train_model(model,train,args)
            acc = test_model(model,test,name)
            meanAcc += acc

    meanAcc = meanAcc * 1.0 / time / len(names)
    print('MeanACC:  ',meanAcc)
    return meanAcc
def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')


    parser.set_defaults(model_type='GraphSage',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=32,
                        dropout=0.0,
                        epochs=200,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()
def main():
    args = arg_parse()
    score(args)
if __name__ == '__main__':
    main()
