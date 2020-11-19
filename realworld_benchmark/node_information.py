import networkx

def get_nodes_degree(graph):
    return graph.in_degrees()

def get_nodes_closeness_centrality(graph):
    return networkx.closeness_centrality(graph.to_networkx())

def get_nodes_betweenness_centrality(graph):
    return networkx.betweenness_centrality(graph.to_networkx())