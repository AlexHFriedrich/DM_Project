#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkit as nk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# file formats
FORMATS =  {'KONECT': nk.graphio.Format.KONECT, 
            'SNAP': nk.graphio.Format.SNAP, 
            }


def plot_graph(Gnk, title, outfile): 
    """plot networkit graph, with nodes colored by degree
    
    :param Gnk: a networkit graph
    :type Gnk: networkit.graph
    :param title: title of figure
    :type title: str
    :param outfile: name of output file
    :type outfile: str    
    """

    
    # convert networkit graph to networkx graph for plotting
    Gnx = nk.nxadapter.nk2nx(Gnk) 
    
    # get node degrees for coloring
    degrees = Gnx.degree() 
    nodes = Gnx.nodes()
    n_color = np.array([degrees[n] for n in nodes])
    
    # calculate spring layout
    pos = nx.spring_layout(Gnx)
    
    # plot graph
    fig, ax = plt.subplots(figsize = (9,9))
    fig.suptitle(title, fontsize = 15, fontweight ='bold')
    sc = nx.draw_networkx(Gnx, pos = pos, with_labels = False,
                     node_size=1,  node_shape='o',
                     edge_color = 'grey', width = 0.1,
                     nodelist=nodes, node_color=n_color, cmap='magma', 
                     vmin = 0, vmax = max(n_color), 
                     arrowsize = 2, ax = ax)
    ax.grid(False)
    plt.savefig(outfile)
    plt.close()
    
    
    
def get_subgraph(Gnk, n):
    """get subgraph of top n nodes based on (out)degree and their (out)neighbors
    
    :param Gnk: a networkit graph
    :type Gnk: networkit.graph
    :param n: use top n nodes
    :type n: int
    :return: subgraph 
    :rtype: networkit.graph
    """
    
    #calculate degree
    d = nk.centrality.DegreeCentrality(Gnk, outDeg=True)
    d.run()    
    # get top n nodes
    top_nodes = [x[0] for x in d.ranking()[0:n]]
    # get subgraph with top n nodes and (out) neighbors
    subgraph = nk.graphtools.subgraphAndNeighborsFromNodes(Gnk, top_nodes, 
                                                           includeOutNeighbors=True)
    return subgraph


def print_stats(Gnk): 
    """print basic stats of networkit graph
    
    :param Gnk: a networkit graph
    :type Gnk: networkit.graph
    """
    
    dout = np.array(nk.centrality.DegreeCentrality(Gnk, outDeg = True).run().scores())
    din = np.array(nk.centrality.DegreeCentrality(Gnk, outDeg = False).run().scores())
    print('Number of nodes: {}'.format(Gnk.numberOfNodes()))
    print('Number of edges: {}'.format(Gnk.numberOfEdges()))
    print('Number of self loops: {}'.format(Gnk.numberOfSelfLoops()))
    print('max out-degree: {}'.format(max(dout)))
    print('max in-degree: {}'.format(max(din)))
    print('directed: {}'.format(Gnk.isDirected()))
    print('weighted: {}\n\n'.format(Gnk.isWeighted()))



if __name__ == '__main__':
    n = 30
    
    # file format
    file_format = 'KONECT'
    
    # path to files  
    files = {'Yeast':'../Data/moreno_propro/out.moreno_propro_propro',
             'Adolescent Health': '../Data/moreno_health/out.moreno_health_health'}   
    
    
    for graph_name, file in files.items(): 
        
        # read graph
        Gnk = nk.graphio.readGraph(file, fileformat = FORMATS[file_format])
        
        # print basic stats of graph
        print(graph_name)
        print_stats(Gnk)
        
        # plot full graph
        plot_graph(Gnk, graph_name, './plots/{}_graph.pdf'.format(graph_name)) 
        
        # get subgraph including n top (out) degree nodes and (out) neighbors 
        subgraph = get_subgraph(Gnk, n)
        
        # plot subgraph
        plot_graph(subgraph,
                   '{}\n {} top degree nodes and neighbors'.format(graph_name, n), 
                   './plots/{}_top{}graph.pdf'.format(graph_name, n))