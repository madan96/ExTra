import argparse
import os
import sys
import time

from qlearn import train_softmax, test
from bisim_transfer.bisimulation import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--mode',
        default='train',
        type=str
    )
    argparser.add_argument(
        '--env-name',
        default='FourLargeRooms',
        type=str
    )
    argparser.add_argument(
        '--alpha',
        default=0.2,
        type=float
    )
    argparser.add_argument(
        '--epsilon',
        default=0.1,
        type=float
    )
    argparser.add_argument(
        '--discount',
        default=0.99,
        type=float
    )
    argparser.add_argument(
        '--temp',
        default=1,
        type=float
    )
    argparser.add_argument(
        '--num-iters',
        default=1000,
        type=int
    )
    argparser.add_argument(
        '--num-seeds',
        default=10,
        type=int
    )
    argparser.add_argument(
        '--policy-dir',
        default='saved_qvalues/optimal_qvalues',
        type=str
    )
    argparser.add_argument(
        '--transfer',
        default='lax',
        type=str
    )
    argparser.add_argument(
        '--src-env',
        default='FourSmallRooms_11',
        type=str
    )
    argparser.add_argument(
        '--tgt-env',
        default='FourLargeRooms',
        type=str
    )
    argparser.add_argument(
        '--solver',
        default='pyemd',
        type=str
    )
    argparser.add_argument(
        '--lfp-iters',
        default=5,
        type=int
    )
    argparser.add_argument(
        '-th',
        '--threshold',
        default=0.01,
        type=float
    )
    argparser.add_argument(
        '-dfk',
        '--discount-kd',
        default=0.9,
        type=float
    )
    argparser.add_argument(
        '-dfr',
        '--discount-r',
        default=0.1,
        type=float
    )
    argparser.add_argument(
        '-ma', '--match-action',
        action='store_true',
        dest='debug',
        help='Match actions with ground truths and generate plots'
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    if args.transfer == 'basic':
        bisimulation = LaxBisimulation(args)
    elif args.transfer == 'lax':
        bisimulation = LaxBisimulation(args)
    elif args.transfer == 'pess':
        bisimulation = PessBisimulation(args)
    elif args.transfer == 'optimistic':
        bisimulation = OptBisimulation(args)
    else:
        raise ValueError("Provide a valid transfer metric")
        
    start = time.time()
    if args.mode == 'train':
        if (os.path.isfile('transfer_logs/Dist-sa_' + args.src_env + '_' + args.tgt_env + '.npy')
            and os.path.isfile('transfer_logs/Dist-matrix_' + args.src_env + '_' + args.tgt_env + '.npy')):
            bisimulation.d_sa_final = np.load('transfer_logs/Dist-sa_' + args.src_env + '_' + args.tgt_env + '.npy')
            bisimulation.dist_matrix_final = np.load('transfer_logs/Dist-matrix_' + args.src_env + '_' + args.tgt_env + '.npy')
        else:
            bisimulation.execute_transfer()
        train_softmax.train(bisimulation, args)
    else:
        test.test(args)
    end = time.time()
    print ("Time taken: ", end - start)