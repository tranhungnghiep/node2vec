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

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default node2vec is 10. Default deepwalk is 5. Default gensim is 10. W2V is 5, 15-20 shows good performance in some cases (node2vec paper). Complexity of w2v increases linearly. \n\
                            Should try [10] or 20.')

    parser.add_argument('--negative-sample', type=int, default=10,
                        help='Number of negative samples. Default gensim is 5. Complexity of w2v increases linearly? \n\
                            Should try 5 or [10].')

    parser.add_argument('--iter', type=int, default=10,
                        help='Number of epochs over the corpus. Default is 1. Default gensim is 5. More is better, complexity of w2v increases linearly, but optimization time is small part. \n\
                            Should try 5 or [10].')

    parser.add_argument('--workers', type=int, default=60,
                        help='Number of parallel workers. Default node2vec is 8. Default gensim is 3. More threads than cores is a typical technique to speed up. \n\
                            Should use [60] workers on 32 cores. (Should also run many instances of node2vec for many networks.)')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted',
                        action='store_false')  # Luckily correct here, but the code is wrong. Should use same dest.
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')  # Same error here.
    parser.set_defaults(directed=False)

    parser.add_argument('--root-path-input',
                        help='Root path input. Default: None.')

    parser.add_argument('--root-path-output',
                        help='Root path output. Default: None.')

    parser.add_argument('--mag-file', type=int, default=0,
                        help='Specify what MAG network files to process. Default 0: use default input/output files.')

    parser.add_argument('--test-year', type=int, default=1996,
                        help='Test year. Default 1996.')

    largs = parser.parse_args()

    # TEST:
    # largs.weighted = True
    largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate.edgelist'
    largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate.emb'
    # largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate_str.edgelist'
    # largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate_str.emb'
    # largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate_w.edgelist'
    # largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate_w.emb'
    # largs.input = '/Users/mac/PythonProjects/node2vec/graph/karate_w_str.edgelist'
    # largs.output = '/Users/mac/PythonProjects/node2vec/emb/karate_w_str.emb'

    # AUTOMATE RUNNING:
    print('First try, use (new) default params.')
    largs.root_path_input = '/mnt/storage/private/nghiep/Data/MAG/Unzip/CitCount6n'
    largs.root_path_output = '/mnt/storage/private/nghiep/Data/CitationCount/MAG7/Embeddings'
    if largs.mag_file == 1:
        largs.input = os.path.join(largs.root_path_input, 'PAPER_CITATION_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'PAPER_CITATION_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = True
        largs.weighted = False
    elif largs.mag_file == 2:
        largs.input = os.path.join(largs.root_path_input, 'AUTHOR_CITATION_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'AUTHOR_CITATION_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = True
        largs.weighted = True
    elif largs.mag_file == 3:
        largs.input = os.path.join(largs.root_path_input, 'VENUE_CITATION_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'VENUE_CITATION_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = True
        largs.weighted = True
    elif largs.mag_file == 4:
        largs.input = os.path.join(largs.root_path_input, 'PAPER_SHARE_AUTHOR_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'PAPER_SHARE_AUTHOR_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = False
        largs.weighted = True
    elif largs.mag_file == 5:
        largs.input = os.path.join(largs.root_path_input, 'AUTHOR_SHARE_PAPER_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'AUTHOR_SHARE_PAPER_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = False
        largs.weighted = True
    elif largs.mag_file == 6:
        largs.input = os.path.join(largs.root_path_input, 'AUTHOR_SHARE_VENUE_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'AUTHOR_SHARE_VENUE_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = False
        largs.weighted = True
    elif largs.mag_file == 7:
        largs.input = os.path.join(largs.root_path_input, 'VENUE_SHARE_AUTHOR_NETWORK_' + str(largs.test_year) + '.txt')
        largs.output = os.path.join(largs.root_path_output, 'VENUE_SHARE_AUTHOR_EMB_' + str(largs.test_year) + '.txt')
        largs.directed = False
        largs.weighted = True

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
    except TypeError:  # Use string ID.
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
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0,
                     sg=1, negative=args.negative_sample,
                     workers=args.workers, iter=args.iter,
                     seed=7)  # Init and train, use skip-gram with negative sampling.
    try:
        model.save_word2vec_format(args.output)
    except DeprecationWarning:
        model.wv.save_word2vec_format(args.output)  # update code to new version of gensim.

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
Note:
- Each time running produce a very different tsne, which means the embedding is not stable.
    -> make sure it converges: increase all.
"""