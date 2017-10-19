'''
tools for handling graphs with several edge types
'''
from __future__ import division
import numpy as np

def multi_to_stringed(graphs):
    '''
    :param graphs: 2d numpy array where each entry is a flat binary adjacency matrix (NxN)
    :returns: even flatter single graph where multigraph rep is stridden across NxN entries
    '''
    return graphs.reashape((graphs.shape[0] * graphs.shape[1],))

def stringed_to_multi(multigraph, n_edgetypes)
    '''
    :param multigraph: 1d numpy array representing stringed graph types
    :param n_edgetypes: number of graphs superimposed in multigraph
    :returns: 2d numpy array where each entry is a flat binary adjacency matrix (NxN)
    '''
    n_square = multigraph.shape[0] // n_edgetypes
    return multigraph.reshape(n_edgetypes, n_square)

def stringed_to_graph_list(multigraph, n_edgetypes)
    '''
    :param multigraph: 1d numpy array representing stringed graph types
    :param n_edgetypes: number of graphs superimposed in multigraph
    :returns: array of 2d numpy arrays representing a graph each
    '''
    n_square = multigraph.shape[0] // n_edgetypes
    n_nodes = int(np.sqrt(n_square))
    return [G.reshape((n_nodes, n_nodes)) for G in stringed_to_multi(multigraph, n_edgetypes)]

def multi_to_any(graphs):
    '''
    :param graphs: 2d numpy array where each entry is a flat binary adjacency matrix (NxN)
    :returns: single graph where all typed edges for a node pair become a single edge
    '''
    n_nodes = int(np.sqrt(graphs.shape[1]))
    flat_any_G = graphs.max(axis=0)
    return flat_any_G.reshape((n_nodes, n_nodes))
    
def multi_to_all(graphs):
    '''
    :param graphs: 2d numpy array where each entry is a flat binary adjacency matrix (NxN)
    :returns: single graph where node pairs in all input graphs become a single edge
    '''
    n_nodes = int(np.sqrt(graphs.shape[1]))
    flat_all_G = graphs.min(axis=0)
    return flat_all_G.reshape((n_nodes, n_nodes))

def weighted_comb(graphs, weights)
    '''
    :param graphs: 2d numpy array where each entry is a flat binary adjacency matrix (NxN)
    :param weights: list or array of weights corresponding to number of graphs
    :returns: new graph with weighted edges
    '''
    n_nodes = int(np.sqrt(graphs.shape[1]))
    warr = np.array(weights)
    assert graphs.shape[0] == warr.shape[0]
    return warr.dot(graphs).reshape((n_nodes, n_nodes))
    
def list_to_compact(raw_graphs)
    '''
    :param raw_graphs: list of 2d NxN numpy array graphs
    :returns: 2d array of shape (len(raw_graphs), N**2)
    '''
    n_nodes = raw_graphs[0].shape[0]
    return np.array([g.reshape((n_nodes**2,)) for g in raw_graphs])

### multigraph configurations = feature extractors ###

def features_per_graph(MG, n_edgetypes, feat):
    '''
    :param MG: multigraph represented as a single stringed representation
    :param n_edgetypes: number of graphs superimposed in multigraph
    :param feat: feature extraction function that applies to a single graph
    :returns: array of results
    '''
    return [feat(G) for G in stringed_to_graph_list(MG, n_edgetypes)]
    
def multigraph_features_sum(MG, n_edgetypes, feats):
    return sum([sum(features_per_graph(MG, n_edgetypes, f)) for f in feats])
