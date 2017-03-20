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
import ctypes


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
        Call parallel simulate_walks.
        Default using pipe.
        '''

        return simulate_walks_pipe(num_walks, walk_length)
        # return simulate_walks_sharedarray(num_walks, walk_length)

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


def simulate_walks_pipe(num_walks, walk_length):
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
        walks.extend(pool.map(node2vec_walk_parallel_pipe, [(walk_length, node) for node in nodes]))

    pool.close()
    pool.join()

    return walks


def simulate_walks_sharedarray(num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.

    Parallel for each node. Using auto chunksize.
    Note that map is like for-loop, it will call the func many times, and automatically distribute args to different processes.
        So no need to parallel for each num_walk, as num_walks is very small.

    Use shared array of string to collect result from subprocesses. Need to carefully avoid conflict write.

    Not working: cannot pass multiprocessing.Array() in map?
    '''

    global g_G, g_workers
    pool = multiprocessing.Pool(g_workers)

    nodes = list(g_G.nodes())
    walks_array = multiprocessing.Array(ctypes.c_char_p, num_walks * len(nodes) * walk_length, lock=False)

    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter + 1), '/', str(num_walks))
        random.shuffle(nodes)
        pool.map(node2vec_walk_parallel_sharedarray, [(walk_length, node, walks_array, walk_iter, len(nodes), node_position) for node_position, node in enumerate(nodes)])  # Error: ValueError: ctypes objects containing pointers cannot be pickled. TODO: convert string id to int id or use char array. But passing to map is pickling, still slow? Using global var is making copies, cannot collect data?

    pool.close()
    pool.join()

    # return np.frombuffer(walks_array, dtype=int).reshape((-1, walk_length)).tolist()  # This only work for non-pointer shared array. This numpy array use same memory with walks_array.
    return np.array(walks_array[:]).reshape((-1, walk_length)).tolist()  # Be careful here.


def node2vec_walk_parallel_pipe(args):
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


def node2vec_walk_parallel_sharedarray(args):
    '''
    Simulate a random walk starting from start node.
    '''
    global g_G, g_alias_nodes, g_alias_edges

    walk_length = args[0]
    start_node = args[1]
    walks_array = args[2]
    walk_iter = args[3]
    len_nodes = args[4]
    node_position = args[5]

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

    # e.g.:
    # 0*100*50+0*50=0 -> [0:0+50] => ok.
    # 0*100*50+5*50=250 -> [250:300] => ok.
    # 1*100*50+0*50=5000 -> [5000:5050] => ok.
    # 1*100*50+5*50=5250 -> [5250:5300] => ok.
    walks_array[walk_iter * len_nodes * walk_length + node_position * walk_length:walk_iter * len_nodes * walk_length + (node_position+1) * walk_length] = walk  # Be careful here.


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
