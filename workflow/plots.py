#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module for plotting co-expression network analysis results

"""
Created on Tue Aug  3 15:45:26 2021

@author: Camila Riccio
"""

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt

# General configuration of matplotlib
rc('font', family='serif', size=18)
rc('text', usetex=False)

# Default colors
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#7f7f7f', '#bcbd22', '#17becf']


def plot_phenotypic_traits_boxplot(P_control,P_stress):
    
    '''
    Comparative boxplots of the phenotypic traits under control and stress 
    conditions.
    
    :param P_control: Phenotypic data under control condition for m genotypes
        and p traits. Shape (m,p)
    :type P_control: DataFrame
    :param P_stress: Phenotypic data under stress condition for m genotypes
        and p traits. Shape (m,p)
    :type P_control: DataFrame
    '''
    Pc = P_control.copy()
    Ps = P_stress.copy()
    Pc['condition'] = 'control'
    Ps['condition'] = 'stress'
    
    P_all = pd.concat([Pc,Ps])
    
    fig, ax = plt.subplots(1,P_control.shape[1],figsize=(15,5))
    
    traits = P_control.columns.tolist()
    for t in range(P_control.shape[1]):
        
        sns.boxplot(x='condition', y=traits[t], data = P_all, showfliers = False,
                    palette='pastel', ax = ax[t])
        sns.stripplot(x='condition', y=traits[t], data = P_all, jitter=True, 
                      dodge=True, marker='o', alpha=0.5, color='grey', ax = ax[t])
        ax[t].set_xlabel('')
    
    fig.tight_layout()
        
def plot_continuous_degree_distribution(corx,nBreaks=30,ax=None):
    '''
    Create a plot of the degree distribution for a weighted adjacency matrix,
    Fits a linear model between the logarithm of the probability that a node
    belongs to a given strenght interval and the average strength of the nodes
    within each strenght interval.
    
    :param corx: correlation matrix (with 1's in the diagonal)
    :type param: DataFrame
    :param nBreaks: number of bins in strength histogram
    :type nBreaks: int
    
    '''
    from sklearn.linear_model import LinearRegression
    
    k = [sum(x)-1 for x in zip(*corx.values)]
    
    hist, bins = np.histogram(k,bins=nBreaks)
    bin_ids = np.digitize(k, bins[1:-1],right=True) 
    interval_sum = np.bincount(bin_ids,  weights=k) 
    dk = np.divide(interval_sum,hist)
    p_dk = np.divide(hist,len(k))          
    inds = np.where(np.isnan(dk))          
    for i in inds: dk[i] = (bins[i+1]+bins[i])/2

    log_dk = np.array(np.log10([x+1e-09 for x in dk])).reshape((-1,1))
    log_pdk = np.array(np.log10([x+1e-09 for x in p_dk]))
    
    # linear model
    lm = LinearRegression().fit(log_dk,log_pdk)
    log_reg = lm.predict(log_dk)
    
    ax = ax or plt.gca()
    ax.scatter(log_dk,log_pdk,color='k',marker='o')
    ax.plot(log_dk,log_reg,color='r')
    # ax.set_title(r'$R^2$ = {0}'.format(lm.score(log_dk, log_pdk))) # R^2
    ax.set_xlabel(r'$\log k$')
    ax.set_ylabel(r'$\log P(k)$')
    
def plot_scale_free_adjustment(S,A=None,beta=None):
    '''
    Create a comparative plot of the weighted degree distribution for two 
    adjacency matrices. Useful to visualize how the degree distribution of a
    similarity matrix (S) becomes a scale-free degree distribution by raising
    the elements of the similarity matrix to a beta exponent in matrix (A) 
    
    :param S: Similarity matrix of shape (n2,n2) with beta equal to 1
    :type S: DataFrame
    :param A: Adjacency matrix of shape (n2,n2). This parameter is optional
        and it will be ignored if beta is not None.
    :type A: DataFrame
    :param beta: power to which each element of the matrix S is raised 
        to force a scale-free degree distribution.
    :type beta: int
    '''
    if not(beta is None):
        A = S.pow(beta)        
        
    fig, ax = plt.subplots(1,2,figsize=(15,6))
    
    plot_continuous_degree_distribution(S,nBreaks=30,ax=ax[0])
    ax[0].set_title(r'$\beta$ = {0}'.format(1))
    plot_continuous_degree_distribution(A,nBreaks=30,ax=ax[1])
    ax[1].set_title(r'$\beta$ = {0}'.format(beta))
    
    fig.tight_layout()
    
def autolabel1(rects,ax=None,color='k'):
    """
    Attach a text label above each bar displaying its height
    """
    ax = ax or plt.gca()
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '{0}'.format(height),
                ha='center', va='bottom',c=color)
        
def autolabel2(rects,ax=None,color='k'):
    """
    Attach a text label above each bar displaying its height
    """
    ax = ax or plt.gca()
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '{0}%'.format(round(height,2)),
                ha='center', va='bottom',c=color) 
    
def plot_overlapping_histogram(F):
    '''
    Plot histogram with the overlapping percentage of the genes, measured 
    as the proportion of modules to which each gene belongs.
    
    :param F: Affiliation matrix from the HLC clustering. Rows represent genes
        and columns represent modules. An entry in F is equal to 1 if the gene
        belongs to the module, zero otherwise.
    :type F: np.ndarray[bool]
    '''
    
    MxN = F.sum(1) #modules per node
    overlap = [100*(m-1)/(F.shape[1]-1) for m in MxN]
    F_over = [g for g in overlap if g==0]
    T_over = [g for g in overlap if g>0]
    hist, bin_edges = np.histogram(T_over,bins=10,range=(0,max(T_over)))

    hist = [len(F_over)]+ list(hist)
    bins = [0]+list(bin_edges)

    means = []
    for i in range(len(bins)-1):
        means.append(round((bins[i]+bins[i+1])/2,2))

    hist_p = np.array(hist)/sum(hist)
    cum = 0
    hist_p2=[]
    for h in hist_p:
        cum += h
        hist_p2.append(cum)


    hist_p2 = 100*np.array(hist_p2)
    
    fig,ax = plt.subplots(figsize=(16,8))
    barp = ax.bar(range(len(hist)),hist_p2,width=1,facecolor='lightskyblue',edgecolor='black') 
    plt.close()
    
    fig,ax = plt.subplots(figsize=(16,8))

    # Plot the histogram heights against integers on the x axis
    # rect = ax.bar(range(len(hist)),hist,width=1,facecolor='lightblue',edgecolor='black') 
    rect = ax.bar(range(len(hist)),hist,width=1,facecolor='lightskyblue',edgecolor='black') 

    # Set the ticks to the middle of the bars
    ax.set_xticks([i for i,j in enumerate(hist)])

    # Set the xticklabels to a string that tells us what the bin edges were
    labels = ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)]
    labels[0] = 'non-overlapping'
    ax.set_xticklabels(means,rotation=90)
    ax.set_ylim([0, max(hist)+450])
    ax.set_xlabel('% overlapping')
    ax.set_ylabel('number of genes')
    autolabel1(rect)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel('Cumulative percentage of genes', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(hist)), hist_p2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 115])
    autolabel2(barp,ax=ax2,color=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('figures/artificial_modules_percentage_LFC.pdf',dpi=400, bbox_inches='tight', pad_inches=0.02)

def plot_overlapping_pie(F):
    '''
    Create a pie chart of the percentage of genes that belong to zero, one,
    or multiple modules.
    
    :param F: Affiliation matrix from the HLC clustering. Rows represent genes
        and columns represent modules. An entry in F is equal to 1 if the gene
        belongs to the module, zero otherwise.
    :type F: np.ndarray[bool]
    '''
    plt.subplots(figsize=(10,6))
    nbr_node_clusters = F.sum(1)
    
    zero = sum(nbr_node_clusters==0) # genes belong to zero communities
    one = sum(nbr_node_clusters==1) # genes belong to exactly one module
    multiple = sum(nbr_node_clusters>1) # genes belong to multiple modules

    labels = ['zero','one','multiple']
    data = [zero,one,multiple]
    colors = ['tab:blue','tab:orange','tab:green']
    plt.pie(data,labels=labels,colors=colors)

    
    
    
    
    
        


