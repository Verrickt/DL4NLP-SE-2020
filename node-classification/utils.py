import torch.optim as optim

def parse_optimizer(parser):
    pass
def build_optimizer(args,params):
    adam = optim.Adam(lr=args.lr,weight_decay=args.weight_decay,params=params)
    return None,adam
