"""
Parallel sharing by global scope var and staticmethod/classmethod and module level method to be parallelized (note that cannot maintain module global scope var without an instance, and cannot access module level method through class/instance, and cannot pickling static/classmethod => hence the combination of global var + static/classmethod + module level method.).
    -> access global scope var with declaration global in each function.
    -> staticmethod is just like module level method, access through class or instance.
        --> if need to access class attribute, need to hardcode class name, e.g., Graph.something.
    -> classmethod is similar to staticmethod, access through either class or instance (type(instance) is passed).
        --> it's better in case of inheritance: it has the cls argument, so it know what the current class is, e.g., cls.something. => use classmethod when need to access class attribute.
    -> method to be parallelized defined at module level, then called by static/class method.

One other way is sharing by class scope with class var.
    But it's more confusing and error prone:
        -> need to access var through class: type(self).var in instancemethod or cls.var in classmethod, if use self.var it will make a copy of the var for each instance.

Note that class Graph is used just as a package encapsulating all vars/methods for convenience, and it maintains the value of var in module.

Note that share data only works on Linux/Unix os.fork, but without guarantee, so it **may** make copies of data.

Author: Tran Hung Nghiep
"""


import numpy as np
np.random.seed(7)
import networkx as nx
import random
random.seed(7)

import multiprocessing


g_G = None
g_is_directed = None
g_p = None
g_q = None
g_alias_nodes = None
g_alias_edges = None
g_workers = None


class Graph:
    def __init__(self, nx_G, is_directed, p, q, workers):
        global g_G, g_is_directed, g_p, g_q, g_workers
        g_G = nx_G
        g_is_directed = is_directed
        g_p = p
        g_q = q
        g_workers = workers

    @staticmethod
    def simulate_walks(num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.

        Parallel for each node. Using auto chunksize.
        Note that map is like for-loop, it will call the func many times, and automatically distribute args to different processes.
            So no need to parallel for each num_walk, as num_walks is very small.
        '''

        global g_G, g_workers
        pool = multiprocessing.Pool(g_workers)

        walks = []
        nodes = list(g_G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            walks.extend(pool.map(node2vec_walk_parallel, [(walk_length, node) for node in nodes]))

        pool.close()
        pool.join()

        return walks

    @staticmethod
    def preprocess_transition_probs():
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        global g_G, g_is_directed

        global g_alias_nodes
        g_alias_nodes = {}
        for node in g_G.nodes():
            unnormalized_probs = [g_G[node][nbr]['weight'] for nbr in sorted(g_G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            g_alias_nodes[node] = alias_setup(normalized_probs)

        global g_alias_edges
        g_alias_edges = {}
        triads = {}

        if g_is_directed:
            for edge in g_G.edges():
                g_alias_edges[edge] = get_alias_edge(edge[0], edge[1])
        else:
            for edge in g_G.edges():
                g_alias_edges[edge] = get_alias_edge(edge[0], edge[1])
                g_alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0])

        return


def node2vec_walk_parallel(args):
    '''
    Simulate a random walk starting from start node.
    '''
    global g_G, g_alias_nodes, g_alias_edges

    walk_length = args[0]
    start_node = args[1]

    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(g_G.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(g_alias_nodes[cur][0], g_alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next = cur_nbrs[alias_draw(g_alias_edges[(prev, cur)][0],
                                           g_alias_edges[(prev, cur)][1])]
                walk.append(next)
        else:
            break

    return walk


def get_alias_edge(src, dst):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    global g_G, g_p, g_q

    unnormalized_probs = []
    for dst_nbr in sorted(g_G.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(g_G[dst][dst_nbr]['weight'] / g_p)
        elif g_G.has_edge(dst_nbr, src):
            unnormalized_probs.append(g_G[dst][dst_nbr]['weight'])
        else:
            unnormalized_probs.append(g_G[dst][dst_nbr]['weight'] / g_q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
