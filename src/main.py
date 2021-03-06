'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import os
import numpy as np
np.random.seed(7)
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import random
random.seed(7)

import time
import multiprocessing
import node2vec_parallel


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default node2vec is 10. Default deepwalk is 10. More num walk is better, but complexity of walk sampling is high. \n\
                            Should try [20] or 50.')

    parser.add_argument('--walk-length', type=int, default=50,
                        help='Length of walk per source. Default node2vec is 80. Default deepwalk is 40. More walk length is better, but complexity of walk sampling is high. \n\
                            Should try [50] or 100.')

    parser.add_argument('--dimensions', type=int, default=50,
                        help='Number of dimensions. Default node2vec is 128. Default deepwalk is 64. Default gensim is 100. w2v is larger, but in node2vec paper 100 is ok. Complexity of w2v increases linearly? More important training citcount is too slow: need to reduce dimension. \n\
                            Should try [50] or 100.')

    parser.add_argument('--window-size', type=int, default=15,
                        help='Context size for optimization. Default node2vec is 10. Default deepwalk is 5. Default gensim is 10. W2V is 5, 15-20 shows good performance in some cases (node2vec paper). Complexity of w2v increases linearly. \n\
                            Should try [15] or 20.')

    parser.add_argument('--negative-sample', type=int, default=5,
                        help='Number of negative samples. Default gensim is 5. Usually more is better. Complexity of w2v increases linearly? \n\
                            Should try [5] or 10.')

    parser.add_argument('--iter', type=int, default=5,
                        help='Number of epochs over the corpus. Default is 1. Default gensim is 5. More is better, complexity of w2v increases linearly, but optimization time is small part. \n\
                            Should try [5] or 10.')

    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count() - 2,
                        help='Number of parallel workers. Default node2vec is 8. Default gensim is 3. More threads than cores is a typical technique to speed up. \n\
                            Should use 60 workers on 32 cores. (Should also run many instances of node2vec for many networks.). But this is a shared server, so only use cpu count - 2.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')  # Same error here.
    parser.set_defaults(directed=False)

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted',
                        action='store_false')  # Luckily correct here, but the code is wrong. Should use same dest.
    parser.set_defaults(weighted=False)


    parser.add_argument('--root-path-input',
                        help='Root path input. Default: None.')

    parser.add_argument('--root-path-output',
                        help='Root path output. Default: None.')

    parser.add_argument('--mag-file', type=int, default=0,
                        help='Specify what MAG network files to process. Default 0: use default input/output files.')

    parser.add_argument('--test-year', type=int, default=1996,
                        help='Test year. Default 1996.')

    parser.add_argument('--weight-threshold', type=int, default=0,
                        help='Specify what weight threshold of network file to use. Default 0: use default value for mag 7 network.')

    parser.add_argument('--alternative-file', dest='alternative_file', action='store_true',
                        help='Use altenative files, such as NoneCoAuthor, NoiseProne... Default is not use.')
    parser.set_defaults(alternative_file=False)

    parser.add_argument('--parallel-node2vec', dest='parallel_node2vec', action='store_true',
                        help='Parallel preprocessing transition probs and simulating walks or not. Default is not parallel node2vec.')
    parser.add_argument('--no-parallel-node2vec', dest='parallel_node2vec', action='store_false')
    parser.set_defaults(parallel_node2vec=False)


    parser.add_argument('--no-local-test', dest='local_test', action='store_false',
                        help='Do not use test param value (for local test). Accept passed param. Default use test param value.')
    parser.set_defaults(local_test=True)

    parser.add_argument('--hyperparams', default='first',
                        help='Hyperparams settings. First try: default (small) settings; Second try: use larger hyperparams. Default: first.')

    parser.add_argument('--config', default='normal',
                        help='Network config. Normal: default config; undirect123: networks 1 2 3 as undirected. Default: normal.')

    largs = parser.parse_args()

    # TEST:
    if largs.local_test:
        # largs.directed = True
        # largs.weighted = True
        largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate.edgelist'
        largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate.emb'
        # largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate_str.edgelist'
        # largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate_str.emb'
        # largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate_w.edgelist'
        # largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate_w.emb'
        # largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate_w_str.edgelist'
        # largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate_w_str.emb'

        # largs.parallel_node2vec = True

    # AUTOMATE RUNNING:
    if largs.mag_file != 0:
        largs.root_path_input = '/mnt/storage/private/nghiep/Data/MAG/Unzip/CitCount'
        largs.root_path_output = '/mnt/storage/private/nghiep/Data/CitationCount/MAG/Embeddings'

        if largs.hyperparams == 'first':
            print('First try, use (new) default params.')
            largs.root_path_output = os.path.join(largs.root_path_output, 'MAG7')
        elif largs.hyperparams == 'second':
            print('Second try, use larger params.')
            largs.root_path_output = os.path.join(largs.root_path_output, 'MAG7L')
            largs.num_walks = 50
            largs.walk_length = 100
            largs.dimensions = 100
            largs.window_size = 20
            largs.negative_sample = 10
            largs.iter = 10

        if largs.mag_file == 1:
            if largs.config == 'normal':
                largs.directed = True
                largs.weighted = False
                largs.root_path_output = os.path.join(largs.root_path_output, 'direct123')
            elif largs.config == 'undirect123':
                largs.directed = False
                largs.weighted = False
                largs.root_path_output = os.path.join(largs.root_path_output, 'undirect123')
            largs.input = os.path.join(largs.root_path_input, 'PAPER_CITATION_NETWORK_' + str(largs.test_year) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'PAPER_CITATION_EMB_' + str(largs.test_year) + '.txt')
        elif largs.mag_file == 2:
            if largs.config == 'normal':
                largs.directed = True
                largs.weighted = True
                largs.root_path_output = os.path.join(largs.root_path_output, 'direct123')
            elif largs.config == 'undirect123':
                largs.directed = False
                largs.weighted = True
                largs.root_path_output = os.path.join(largs.root_path_output, 'undirect123')
            if largs.weight_threshold == 0:
                largs.weight_threshold = 2
            largs.input = os.path.join(largs.root_path_input, 'AUTHOR_CITATION_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'AUTHOR_CITATION_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
        elif largs.mag_file == 3:
            if largs.config == 'normal':
                largs.directed = True
                largs.weighted = True
                largs.root_path_output = os.path.join(largs.root_path_output, 'direct123')
            elif largs.config == 'undirect123':
                largs.directed = False
                largs.weighted = True
                largs.root_path_output = os.path.join(largs.root_path_output, 'undirect123')
            if largs.weight_threshold == 0:
                largs.weight_threshold = 2
            largs.input = os.path.join(largs.root_path_input, 'VENUE_CITATION_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'VENUE_CITATION_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
        elif largs.mag_file == 4:
            largs.directed = False
            largs.weighted = True
            if largs.weight_threshold == 0:
                largs.weight_threshold = 2
            largs.input = os.path.join(largs.root_path_input, 'PAPER_SHARE_AUTHOR_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'PAPER_SHARE_AUTHOR_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
        elif largs.mag_file == 5:
            largs.directed = False
            largs.weighted = True
            if largs.weight_threshold == 0:
                largs.weight_threshold = 2
            largs.input = os.path.join(largs.root_path_input, 'AUTHOR_SHARE_PAPER_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'AUTHOR_SHARE_PAPER_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
        elif largs.mag_file == 6:
            largs.directed = False
            largs.weighted = True
            if largs.weight_threshold == 0:
                largs.weight_threshold = 5
            largs.input = os.path.join(largs.root_path_input, 'AUTHOR_SHARE_VENUE_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'AUTHOR_SHARE_VENUE_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            if largs.alternative_file:
                largs.input = os.path.join(largs.root_path_input, 'NoneCoAuthor', 'AUTHOR_SHARE_VENUE_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
                largs.output = os.path.join(largs.root_path_output, 'NoneCoAuthor', 'AUTHOR_SHARE_VENUE_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
        elif largs.mag_file == 7:
            largs.directed = False
            largs.weighted = True
            if largs.weight_threshold == 0:
                largs.weight_threshold = 2
            largs.input = os.path.join(largs.root_path_input, 'VENUE_SHARE_AUTHOR_NETWORK_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')
            largs.output = os.path.join(largs.root_path_output, 'VENUE_SHARE_AUTHOR_EMB_' + str(largs.test_year) + '_' + str(largs.weight_threshold) + '.txt')

    if not os.path.isdir(largs.root_path_output):
        os.makedirs(largs.root_path_output)

    return largs


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    try:
        if args.weighted:
            G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
    except TypeError:  # Use Node type str.
        if args.weighted:
            G = nx.read_edgelist(args.input, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''

    try:
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0,
                         sg=1, negative=args.negative_sample,
                         workers=args.workers, iter=args.iter,
                         seed=7)  # Init and train, use skip-gram with negative sampling.
    except TypeError:  # Node type: int convert to str.
        if 1+1 == 0 and len(walks) > 10**6:  # Small size is not worth parallel convert int to str.
            pool = multiprocessing.Pool(args.workers)
            walks = pool.map(mapstr, walks)
            pool.close()
            pool.join()
        else:
            walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0,
                         sg=1, negative=args.negative_sample,
                         workers=args.workers, iter=args.iter,
                         seed=7)  # Init and train, use skip-gram with negative sampling.

    try:
        model.save_word2vec_format(args.output)
    except DeprecationWarning:  # Update code to new version of gensim.
        model.wv.save_word2vec_format(args.output)

    return


def mapstr(walk):
    """
    Helper function for parallel convert node type from int to str.
    :param walk: list of int.
    :return: list of str.
    """

    return map(str, walk)


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    print('START.')
    start_time_main = time.time()  # in second.

    base_time = time.time()
    nx_G = read_graph()
    print('Read graph time: ' + str(time.time() - base_time)); base_time = time.time()

    if False and args.parallel_node2vec:
        G = node2vec_parallel.Graph(nx_G, args.directed, args.p, args.q, args.workers)
    else:
        G = node2vec.Graph(nx_G, args.directed, args.p, args.q)

    G.preprocess_transition_probs()
    print('Preprocess transition probability time: ' + str(time.time() - base_time)); base_time = time.time()

    walks = G.simulate_walks(args.num_walks, args.walk_length)
    print('Simulate walks ' + '(Parallel=' + str(False and args.parallel_node2vec) + ') time : ' + str(time.time() - base_time)); base_time = time.time()

    learn_embeddings(walks)
    print('Compute w2v embeddings time: ' + str(time.time() - base_time)); base_time = time.time()

    print('FINISH.')
    stop_time_main = time.time()
    print('Time (s): ' + str(stop_time_main-start_time_main))


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
Note:
- Each time running produce a very different tsne, which means the embedding is not stable.
    -> make sure it converges: increase walklength, numwalk to let random walks capture more info of network, increase dimension to let embedding capture more info of random walks, increase interation to let embedding converge.
- Node2vec walk sampling is not parallel: try using deepwalk?
    -> code deepwalk is messy: no weighted, directed is not sure, parallel walk sampling seems just a trick using an online generator of random walks.
    => 2 ways to parallelize random walk:
        1. generator trick using 'yield': not sure it would work: check/test.
            -> generator is stateful, it is sequential.
                The only way to parallelize generator is to use multiple iterators for multiple process, and control the starting point of each generator.
                And slow iterator/generator is a known issue with parallel gensim w2v.
                => cannot use generator to parallelize.
        2. => the only way is parallel sampling walks, may use map (with chunk size), may use apply_async?
            -> parallel is too complicated. because cannot share instance data between processes. => stop.

- A problem with parallel node2vec:
    * Result are returned through a pipe, the pipe's size is very small and slow.
        -> when returning big result, it becomes the bottleneck and process goes to sleep.
            => need to use shared memory directly, as in Java.
                -> Python has some supports for this: shared ctypes with multiprocessing.Array(lock=False), multiprocessing.sharedctypes.RawArray
                (Note that Manager() is not directly shared memory, it is a central process memory, it needs to serialize and deserialize (pickle/unpickle) object to send between process: so it's also slow.)
    * Moreover, global vars are copied even though subprocesses do not modify it.
        This is because refcount of the vars are changed.
==> parallel processing on python is a failure.
"""