#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkit as nk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
import time
import pandas as pd

FORMATS =  {'KONECT': nk.graphio.Format.KONECT, 
            'SNAP': nk.graphio.Format.SNAP, 
            }


# centrality measure functions from networkx
centr_func = {'Degree Centrality': nx.degree, 
              'Betweenness centrality': nx.betweenness_centrality, 
              'Eigenvector centrality':nx.eigenvector_centrality
             }

# arguments needed to include weight in centrality measure calculation 
weighted_args = {'Degree Centrality': {'weight':'weight'}, 
                 'Betweenness centrality': {'weight':'weight'} , 
                 'Eigenvector centrality':{'weight':'weight', 'max_iter':1000}
                }

# arguments to not include weight in centrality measure calculation 
unweighted_args = {'Degree Centrality': {'weight':None}, 
                   'Betweenness centrality': {'weight':None} , 
                   'Eigenvector centrality':{'weight':None, 'max_iter':1000}
                  }
             

def plot_hist(data, title, outname):
    """ function for plotting centrality measure histograms"""
    
    fontsize = 9
    nmeasures = len(data)
    fig, axes = plt.subplots(1, nmeasures, figsize = (6.9, 3))
    fig.suptitle(title, fontweight = 'bold')
    
    col = 0
    for measure, scores in data.items(): 
        ax = axes[col]
        values = list(scores.values())
        if measure == 'Degree Centrality':
            mindeg = 0
            maxdeg = int(max(values))
            bins  = np.linspace(mindeg-0.5, maxdeg+0.5, maxdeg-mindeg+2)
        else: 
            bins = 50
            
        ax.hist(values, bins = bins, lw = 0.0)
        ax.set_title(measure, fontsize = fontsize+2, fontweight ='bold')
        ax.set_xlabel(measure, fontsize = fontsize)
        ax.set_ylabel('frequency', fontsize = fontsize)
        minimum = min(values)
        maximum = max(values)
        mean = np.mean(values)

        s = 'N={}\nmin = {}\nmean = {}\nmax ={}'.format(len(values), 
                                                        round(minimum, 3), 
                                                        round(mean, 3), 
                                                        round(maximum, 3), )
        ax.text(s = s, x = 0.98, y = 0.98, 
                ha = 'right', va = 'top',
                fontsize = 9,
                transform = ax.transAxes)
        col+=1
        
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

def add_random_weight(Gnx, low=1, high=21):
    """ function for adding random weight to networkx graph"""

    # iterate over all edges
    for edge in Gnx.edges(): 
        u, v = edge
        
        # select random weight (int) between low and high
        weight = np.random.randint(1,21)
        
        # add weight to edge
        Gnx[u][v]['weight'] = weight
        
    return Gnx


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
        
        # convert to networkx graph 
        Gnx = nk.nxadapter.nk2nx(Gnk)
    
        # if not weighted add random weight
        if not weighted: 
            add_random_weight(Gnx)
    
    
        kendalls_tau[network_name] = {}
        running_time[network_name  + ' weighted'] = {}
        running_time[network_name  + ' unweighted'] = {}
        data[network_name] = {}
        
        
        
        # calculate centrality measures
        for measure, func in centr_func.items(): 
            running_time[network_name  + ' weighted'][measure]= {}
            running_time[network_name  + ' unweighted'][measure]= {}
            
            
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
            
            
            # store scores for plotting
            if weighted: 
                data[network_name][measure] = dict(weighted_scores)
            else: 
                data[network_name][measure] = dict(unweighted_scores)
    
        # plot histogram of centrality measures
        plot_hist(data[network_name], network_name, network_name+'_hist.pdf')
       
       
    # print tables
    df_kendalltau = pd.DataFrame(data = kendalls_tau)
    df_running_time = pd.DataFrame(data = running_time)
    print('Kendalls tau coefficient between rankings for weighted and unweighted graphs')
    print(df_kendalltau.to_string())
        
    print('Running time [s]')
    print(df_running_time.to_string())
        
    
    
