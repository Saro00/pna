import networkx


def get_nodes_degree(graph):
    return graph.in_degrees()

def get_nodes_closeness_centrality(graph):
    return list(networkx.closeness_centrality(graph.to_networkx()).values())

def get_nodes_betweenness_centrality(graph):
    return list(networkx.betweenness_centrality(graph.to_networkx()).values())

def get_nodes_pagerank(graph):
    return list(networkx.algorithms.link_analysis.pagerank_alg.pagerank(graph.to_networkx()).values())

NODE_INFORMATION = {'degree' : get_nodes_degree, 'closeness_centrality' : get_nodes_closeness_centrality,
                    'betweenness_centrality' : get_nodes_betweenness_centrality, 'pagerank' : get_nodes_pagerank}