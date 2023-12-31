import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str,default='.\dataset_zhy',
                    help='the path of dataset.')
# parser.add_argument('--scenario', type=str,choices=["in", "out"],default='in',
#                     help="the channel scenario")
parser.add_argument('-b', '--batch-size', type=int,metavar='N',default='200',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N',default='0',
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',default=2500,
                    help='number of total epochs to run')
parser.add_argument('--rou', type=str, default=0.01,help='balance coefficient'
                    )
parser.add_argument('--phase',type=int,default=1,choices=[1,2])

args = parser.parse_args()
