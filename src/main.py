'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
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

    # parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
    #                     help='Input graph path')
    #
    # parser.add_argument('--output', nargs='?', default='emb/karate.emb',
    #                     help='Embeddings path')

    # parser.add_argument('--dimensions', type=int, default=100,
    #                     help='Number of dimensions. Default is 128.')
    #
    # parser.add_argument('--walk-length', type=int, default=80,
    #                     help='Length of walk per source. Default is 80.')
    #
    # parser.add_argument('--num-walks', type=int, default=10,
    #                     help='Number of walks per source. Default is 10.')
    #
    # parser.add_argument('--window-size', type=int, default=10,
    #                     help='Context size for optimization. Default is 10.')
    #
    # parser.add_argument('--iter', default=1, type=int,
    #                     help='Number of epochs in SGD. Default is 1.')

    # parser.add_argument('--workers', type=int, default=8,
    #                     help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted',
                        action='store_false')  # Luckily correct here, but the code is wrong. Should use same dest.
    # parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')  # Same error here.
    parser.set_defaults(directed=False)

    # TEST:
    parser.set_defaults(weighted=False)
    # parser.set_defaults(weighted=True)

    parser.add_argument('--input', nargs='?', default='/Users/mac/PythonProjects/node2vec/graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='/Users/mac/PythonProjects/node2vec/emb/karate.emb',
                        help='Embeddings path')
    #
    # parser.add_argument('--input', nargs='?', default='/Users/mac/PythonProjects/node2vec/graph/karate_str.edgelist',
    #                     help='Input graph path')
    #
    # parser.add_argument('--output', nargs='?', default='/Users/mac/PythonProjects/node2vec/emb/karate_str.emb',
    #                     help='Embeddings path')

    # parser.add_argument('--input', nargs='?', default='/Users/mac/PythonProjects/node2vec/graph/karate_w.edgelist',
    #                     help='Input graph path')
    #
    # parser.add_argument('--output', nargs='?', default='/Users/mac/PythonProjects/node2vec/emb/karate_w.emb',
    #                     help='Embeddings path')
    #
    # parser.add_argument('--input', nargs='?', default='/Users/mac/PythonProjects/node2vec/graph/karate_w_str.edgelist',
    #                     help='Input graph path')
    #
    # parser.add_argument('--output', nargs='?', default='/Users/mac/PythonProjects/node2vec/emb/karate_w_str.emb',
    #                     help='Embeddings path')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default node2vec is 10. Default deepwalk is 10. More num walk is better, but complexity of walk sampling is high. \n\
                            Should try [20].')

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
                        help='Number of parallel workers. More threads than cores is a typical technique to speed up. \n\
                            Should use [60] workers on 32 cores.')

    return parser.parse_args()


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