import datasets
import argparse
import torch
import train
import tcnn
import sys
import torchtext.data as data



def main(args):
    torch.manual_seed(args.seed)
    TEXT = data.Field(lower=True)
    LABEL = data.LabelField()
    train_iter,valid_iter,test_iter = datasets.imdb(TEXT,LABEL,args=args)

    args.vocabulary_size = len(TEXT.vocab)
    args.output_dim = len(LABEL.vocab)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    model = tcnn.Text_CNN(args)
    if args.cuda:
        model.cuda()
    if args.snapshot is not None:
        train.evaluate_model(model,test_iter,args)
    else:
        train.train(train_iter,valid_iter,model,args)
        train.evaluate_model(model,test_iter,args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='./model', help='where to save the snapshot')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data 
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=int,nargs='+', default=[3,4,5], help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
    parser.add_argument('-output_dim',type=int,default=2,help='number of output dimension')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default='./model/best_steps_6800.pt', help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-seed',type=int,default=0,help='seed for RNG')
    parser.add_argument('-max_vocabulary_size',type=int,default=10000,help='The maximum size of vocabulary')
    #try:
    #    args = parser.parse_args()
    #    print("====================")
    #    print(args)
    #    print("====================")
    #    main(args)
    #except:
    #    parser.print_help()
    #    sys.exit(0)

    args = parser.parse_args()
    print("====================")
    print(args)
    print("====================")
    main(args)
    parser.print_help()
    sys.exit(0)
