#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkit as nk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.stats import kendalltau
import time
import pandas as pd

# helper functions for easy iteration 
def in_degree(GnxDiGraph, weight = None): 
    return GnxDiGraph.in_degree(weight = weight)

def out_degree(GnxDiGraph, weight = None): 
    return GnxDiGraph.out_degree(weight = weight)



FORMATS =  {'KONECT': nk.graphio.Format.KONECT, 
            'SNAP': nk.graphio.Format.SNAP, 
            }

# centrality measure functions from networkx
centr_func = {'Degree Centrality': nx.degree, 
              'In-Degree Centrality': in_degree, 
              'Out-Degree Centrality': out_degree,
              'Betweenness centrality': nx.betweenness_centrality, 
              'Eigenvector centrality':nx.eigenvector_centrality
             }


# arguments needed to include weight in centrality measure calculation 
# as for betweenness centrality the weight is interpreted as distance, the inverse
# of the weight is chosen for this.
weighted_args = {'Degree Centrality': {'weight':'weight'}, 
                 'In-Degree Centrality': {'weight':'weight'}, 
                 'Out-Degree Centrality': {'weight':'weight'}, 
                 'Betweenness centrality': {'weight':'inverse_weight'} , 
                 'Eigenvector centrality':{'weight':'weight', 'max_iter':1000}
                }

# arguments to not include weight in centrality measure calculation 
unweighted_args = {'Degree Centrality': {'weight':None}, 
                   'In-Degree Centrality': {'weight':None}, 
                   'Out-Degree Centrality': {'weight':None}, 
                   'Betweenness centrality': {'weight':None} , 
                   'Eigenvector centrality':{'weight':None, 'max_iter':1000}
                  }

def write_stats(ax, data, x, y, fontsize): 
    """Writes basic statistics to axis.
    
    :param ax: axis
    :type ax: matplotlib.pyplot.axis
    :param data: data
    :type data: list
    :param x: x position of text to write, in axis coordinates
    :type x: float
    :param y: y position of text to write, in axis coordinates
    :type y: float
    :param fontsize: fontsize of text
    :type fontsize: float
    """     
    
    minimum = min(data)
    maximum = max(data)
    mean = np.mean(data)

    s = 'min = {}\nmean = {}\nmax ={}'.format(round(minimum, 3), 
                                              round(mean, 3), 
                                              round(maximum, 3), )
    ax.text(s = s, x = x, y = y, 
            ha = 'right', va = 'top',
            fontsize = fontsize, transform = ax.transAxes)              


def plot_hist(data, title, outname):
    """plotting centrality measure histograms
    
    :param data: dictionary of dictionaries containing scores for nodes
    :type data: dict
    :param title: title of figure
    :type title: str
    :param outname: name of output file
    :type outname: str
    """
    
    fontsize = 8
    
    # get number of columns and rows to plot 
    nmeasures = len(data)
    ncols = 3
    nrows = int(np.ceil(nmeasures/ncols))
    
    # define figure
    fig, axes = plt.subplots(nrows, ncols, figsize = (6.9, 0.5 + 2*nrows))
    axes = axes.reshape(-1)
    
    # set suptitle of figure
    fig.suptitle(title + ' (N={})'.format(len(list(data.values())[0])), fontweight = 'bold')
    
    # plot histograms for all centrality measures
    axindex = 0
    for measure, scores in data.items():
        
        ax = axes[axindex]
        values = list(scores.values())
        
        if measure == 'Degree Centrality':
            mindeg = 0
            maxdeg = int(max(values))
            bins  = np.linspace(mindeg-0.5, maxdeg+0.5, maxdeg-mindeg+2)
        else: 
            bins = 50
            
        ax.hist(values, bins = bins, lw = 0.0,)
        
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)
        write_stats(ax, values, x=0.98, y=0.98, fontsize = fontsize-1)  
        ax.set_title(measure, fontsize = fontsize+2, fontweight ='bold')
        ax.set_xlabel(measure, fontsize = fontsize)
        ax.set_ylabel('frequency', fontsize = fontsize)
        
        axindex+=1
        
    # remove unused axes
    for i in range(axindex, len(axes)): 
        axes[i].remove()  
        
    # save figure    
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    
    
def write_results(Gnx, scores, network_name):
    """Writes edges and scores for nodes to files for import in Cytoscape
    
    :param Gnx: a networkx graph
    :type Gnx: networkx.Graph or networkx.DiGraph
    :param scores: dictionary of dictionaries containing scores for nodes
    :type scores: dict
    :param network_name: Name of network
    :type network_name: str
    """
    
    with open(network_name+'_edges.txt', 'w') as f: 
        f.write('node1\tnode2\n')
        for edge in Gnx.edges:
            n1, n2 = edge
            f.write('{}\t{}\n'.format(n1, n2))
            
    keys = scores.keys()
    with open(network_name+'_centrality.txt', 'w') as f:  
        f.write('node\t'+'\t'.join(keys)+'\n')
        for node in Gnx.nodes: 
            data = [str(node)] + [str(scores[key][node]) for key in keys]
            f.write('\t'.join(data)+'\n')  
            

def add_random_weight(Gnx, low=1, high=21):
    """Adds random integer weights to networkx graph in interval [low, high).
    
    :param Gnx: a networkx graph
    :type Gnx: networkx.Graph or networkx.DiGraph
    :param low: lowest weight to be drawn, defaults to 1
    :type low: int, optional
    :param high: one above the largest integer to be drawn, defaults to 21
    :type high: int, optional
    """
    
    # set seed for reproducibility
    np.random.seed(0)

    # iterate over all edges
    for edge in Gnx.edges(): 
        u, v = edge
        
        # select random weight (int) between low and high
        weight = np.random.randint(1,21)
        
        # add weight to edge
        Gnx[u][v]['weight'] = weight
        
        
def add_inverse_weight(Gnx):
    """Adds inverse of edge weights as attribute to networkx graph.
    
    :param Gnx: a networkx graph
    :type Gnx: networkx.Graph or networkx.DiGraph
    """
    
    #iterate over all edges
    for edge in Gnx.edges(): 
        u, v = edge
        
        # get weight for current edge
        weight = Gnx[u][v]['weight'] 
        
        # add inverse weight to edge attributes
        Gnx[u][v]['inverse_weight'] = 1/weight 
        

if __name__ == '__main__':
    
    # file format
    file_format = 'KONECT'
    
    # path to files   
    files = {'Yeast':'data/moreno_propro/out.moreno_propro_propro',
             'Adolescent Health': 'data/moreno_health/out.moreno_health_health'}   
    
    # dictionaries to store results
    kendalls_tau = {}
    running_time = {}
    data = {}
    
    
    
    for network_name, file in files.items(): 
        
        # read graph
        Gnk =  nk.graphio.readGraph(file, fileformat = FORMATS[file_format])
        
        # check if graph  weighted
        weighted = Gnk.isWeighted()
        
        # check if graph directed
        directed = Gnk.isDirected()
        
        # convert to networkx graph 
        Gnx = nk.nxadapter.nk2nx(Gnk)
    
        # if not weighted add random weight
        if not weighted: 
            add_random_weight(Gnx)
    
        # add inverse weights for calculation of Betweenness centrality
        add_inverse_weight(Gnx)
        
        
        kendalls_tau[network_name] = {}
        running_time[network_name  + ' unweighted'] = {}
        running_time[network_name  + ' weighted'] = {}
        data[network_name] = {}
        
        
        
        # calculate centrality measures
        for measure, func in centr_func.items(): 
            
            if measure in ['In-Degree Centrality', 'Out-Degree Centrality'] and not directed: 
                continue
            
            running_time[network_name  + ' unweighted'][measure]= {}
            running_time[network_name  + ' weighted'][measure]= {}
            
            
            # calculate centrality measure using weight
            start = time.time()
            weighted_scores = func(Gnx, **weighted_args[measure])
            end = time.time()
            running_time[network_name  + ' weighted'][measure] = end-start
            
            # calculate centrality measure not using weight
            start = time.time()
            unweighted_scores = func(Gnx, **unweighted_args[measure])
            end = time.time()
            running_time[network_name  + ' unweighted'][measure] = end-start
            
            
            # get scores for nodes
            sw = [weighted_scores[n] for n in Gnx.nodes]
            su = [unweighted_scores[n] for n in Gnx.nodes]
            
            # calculate kendall tau for obtained rankings of weighted and unweighted graph
            tau = kendalltau(sw, su).statistic
            kendalls_tau[network_name][measure] = tau
            
            
            # store scores for plotting (if graph was weighted use weighted)
            if weighted: 
                data[network_name][measure] = dict(weighted_scores)
            else: 
                data[network_name][measure] = dict(unweighted_scores)
    
    
        # plot histogram of centrality measures
        plot_hist(data[network_name], network_name, network_name+'_hist.pdf')
        
        # write results to files
        write_results(Gnx, data[network_name], network_name)
       
    # print tables
    df_kendalltau = pd.DataFrame(data = kendalls_tau)
    df_running_time = pd.DataFrame(data = running_time)
    print('\nKendalls tau coefficient between rankings for weighted and unweighted graphs')
    print(df_kendalltau.to_string())
        
    print('\nRunning time [s]')
    print(df_running_time.to_string())
        
    
    
