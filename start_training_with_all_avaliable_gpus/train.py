import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=32,help='')
parser.add_argument('--learning_rate',type=float,default=0.001,help='')
parser.add_argument('--max_number_of_steps',type=int,default=100,help='')
parser.add_argument('--checkpoint_dir',type=str,default='./checkpoint',help='')
parser.add_argument('--optimizer',type=str,default='sgd',help='')
parser.add_argument('--learning_rate_decay_type',type=str,default='fixed',help='')
args = parser.parse_args()

print('begin to train with config:{}'.format(args))

