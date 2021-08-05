#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# co-expression network analysis workflow
# using overlapping community detection (with HLC)
# and modules selection (with LASSO)
"""
Created on Sun Aug  1 12:14:43 2021

@author: Camila Riccio
"""

# Libraries
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from cdlib import algorithms
from scipy import sparse
from sklearn.decomposition import PCA
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import STAP

# # Plotting, own library
from .plots import *


class workflow:
    '''
    Class for condition-specific co-expression network analysis.
    This class builds a co-expression network from RNA-seq data measured under 
    control and stress conditions, detects modules of co-expressed genes
    from the co-expression matrix, and selects a reduced set of genes relevant
    to the stress response represented by a set of phenotypic traits.'''
    
    # Default constructor
    def __init__(self):
        self.D0 = None          # RNA-seq data (control and stress)
        self.P_control = None   # Phenotypic traits under control
        self.P_stress = None    # Phenotypic traits under stress
        self.Pl = None          # LFC of the phenotypic traits
        self.L1 = None          # LFC of the expression for notable genes
        self.A = None           # Adjacency matrix with scale-free property
        self.F = None           # Affiliation matrix (HLC clustering)
        self.M = None           # Eigengenes matrix
        self.beta = None        # beta power for scale-free adjustment
        
        
    def load_data(self,D0,P_control,P_stress,plot=False):
        '''
        Load sample data
        
        :param D0: RNA-seq data measured under 2 contrasting conditions, namely 
            control and stress, for n0 genes and m genotypes with r replicates
            per genotype. Shape (n0,2mr). 
        :type D0: DataFrame
        :param P_control: Phenotypic data measured under control condition
            for m genotypes and p traits. Shape (m,p).
        :type P_control: DataFrame
        :param P_stress: Phenotypic data measured under stress condition
            for m genotypes and p traits. Shape (m,p).
        :type P_stress: DataFrame
        :param plot: if set to TRUE a plot comparing the phenotypic traits
            under control and stress will be produced.
        :type plot: bool
        '''
        
        self.D0 = D0
        self.P_control = P_control
        self.P_stress = P_stress
        
        if plot:
            print('Creating comparative plot of phenotypic traits...')
            plot_phenotypic_traits_boxplot(P_control,P_stress)
            plt.savefig('test/output/figures/boxplots.png',dpi=400, bbox_inches='tight', pad_inches=0.02)
            
        
        
    def load_precomputed_data(self, Pl=None, L1=None, A=None, 
                              F=None,plot_hlc=False):
        '''
        Load pre-procesed data for m genotypes, n2 genes, p phenotypic traits,
        and c modules.
        
        :param Pl: LFC of the phenotypic traits. Shape (m,p)
        :type Pl: DataFrame
        :param L1: LFC of the expression for notable genes. shape (n2,m)
        :type L1: DataFrame
        :param A: Adjacency matrix with scale-free property. Shape (n2,n2)
        :type A: DataFrame
        :param F: Affiliation matrix from the HLC clustering. Shape (n2,c)
        :type F: scipy.sparse.csr.csr_matrix
        :param plot_hlc: if set to True figures will be created summarizing 
            the overlap of the nodes.
        :type plot_hlc: bool
        '''
        if (self.Pl is None):
            self.Pl = Pl
            print('Pl: ',Pl.shape)
        if (self.L1 is None):
            self.L1 = L1
            print('L1: ',L1.shape)
        if (self.A is None):
            self.A = A
            print('A: ',A.shape)
        if (self.F is None):
            self.F = F.toarray()
            print('F: ',F.shape)
            if plot_hlc:
                print('Creating overlapping module detection figures...')
                plot_overlapping_histogram(self.F)
                plt.savefig('test/output/figures/overlapping_histogram.png',
                            dpi=400, bbox_inches='tight', pad_inches=0.02)
                plot_overlapping_pie(self.F)
                plt.savefig('test/output/figures/overlapping_pie.png',
                            dpi=400, bbox_inches='tight', pad_inches=0.02)
            
    
    def DESeq2(D0):
        '''
        Normalization method used to correct for library size and RNA
        composition bias.
        
        :param D0: RNA-seq data measured under 2 contrasting conditions, namely 
            control and stress, for n0 genes and m genotypes with r replicates
            per genotype. Shape (n0,2mr).
        :type D0: DataFrame
        
        :return: normalized RNA-seq data. Shape (n0,2mr)
        :rtype: DataFrame
        '''
        # step 1: take log of all values
        df_deseq = D0.apply(np.log)
        # step 2: Average each raw
        geometric_average = df_deseq.mean(axis=1)
        # Step 3: Filter out genes with Infinity
        df_deseq = df_deseq[geometric_average!=-np.inf]
        # Step 4: Subtract the average log value from the log(count)
        df_deseq = df_deseq.sub(df_deseq.mean(axis=1), axis=0)
        # Step 5: Calculate the median of the ratios for each sample (column)
        medians = df_deseq.median(axis=0)
        # Step 6: Convert the medians to "normal numbers" to get the final scaling factors for each sample
        scaling_factors = np.exp(medians)
        # Divide the original read counts by the scaling factors
        df_deseq = D0.div(scaling_factors, axis=1)
        return df_deseq
    
    def average_replicates(D1):
        '''
        Average replicates
        
        :param D1: normalized RNA-seq data. Shape (n0,2mr)
        :type D1: DataFrame
        
        :return: normalized RNA-seq data averaged by genotype. Shape (n0,2m)
        :rtype: DataFrame
        '''
        D2 = pd.DataFrame()
        unique_vars = list(set([var[0] for var in D1.columns.str.split('_GSM')]))
        unique_vars = np.sort(unique_vars)
        for var in unique_vars:
            D2[var] = D1.filter(regex='^'+var).mean(axis=1)
        return D2
        
    def filter_low_expression_and_variance(D2,var_th=1.5):
        '''
        Remove genes with low expression and low variance in the normalized
        RNA-seq data. Genes with low expression are considered as those having
        more than 80% samples with values smaller than 10. Genes with low
        variance are considered as those whose difference between upper
        quantile and lower quantile is smaller than a threshold var_th.
        
        :param D2:  normalized RNA-seq data averaged by genotype. Shape (n0,2m)
        :type D2: DataFrame
        :param exp_th: upper threshold for genes with low expression.
        :type exp_th: float
        :param var_th: upper threshold for genes with low variance
        :type var_th: float
        
        :return: normalized RNA-seq data averaged by genotype for genes
            with notable expression and variance. Shape (n1,2m) with n1<=n0.
        :rtype: DataFrame
        '''
        # 1. Remove genes with low expression
        q = np.array(D2.quantile(0.8,axis = 1))
        D3 = D2[q>=10]
        
        # 2. Remove genes with low variance
        uq = D3.quantile(0.75,axis = 1)
        lq = D3.quantile(0.25,axis = 1)
        ratio = np.array([(u+1)/(l+1) for u,l in zip(uq,lq)])
        return D3[ratio>var_th]
    
    def expression_LFC(D3,genotypes):
        '''
        Computes differential expression values as logarithm in base 2 of
        the ratio between expression under stress and expression under control.
        
        :param D3: normalized RNA-seq data averaged by genotype for genes
            with notable expression and variance. Shape (n1,2m).
        :type: DataFrame
        
        :return: Log2 Fold Change of expression data. Shape (n1,m)
        :rtype: DataFrame
        '''
        L0 = pd.DataFrame()
        j = 0
        for i in range(0,D3.shape[1],2):
            L0[genotypes[j]] = D3.iloc[:,i:i+2].apply(lambda x: np.log2((x[1]+1)/(x[0]+1)),axis=1)
            j+=1
        
        return L0
    
    def filter_low_variance_LFC(L0,var_th=0.25):
        '''
        Remove genes with low variance in LFC data
        
        :param L0: LFC of expression data. Shape (n1,m)
        :type L0: DataFrame
        :param var_th: upper threshold for genes with low variance.
        :type var_th: float
        
        :return: LFC data for genes with notable variance. Shape (n2,m)
        :rtype: DataFrame
        '''
        uq = L0.quantile(0.75,axis = 1)
        lq = L0.quantile(0.25,axis = 1)
        ratio = np.array([u-l for u,l in zip(uq,lq)])
        return L0[ratio>var_th]
    
    def data_preprocessing(self,expr_var_th=1.5,lfc_var_th=0.25):
        '''
        Build matrices Pl and L1 representing, respectively, the changes in
        phenotypic values and expression levels between control and stress
        condition.
        
        :param expr_var_th: upper threshold for low expression variance
        :type expr_var_th: float
        :param lfc_var_th: upper threshold for low LFC variance
        :type lfc_var_th: float
        
        :return: Matrix L1 of shape (n2,m) with LFC values for each 
            gene-genotype tuple, and Pl matrix of shape (m,p) with LFC values
            for each trait-genotype tuple.
        :rtype: tuple(DataFrame,DataFrame)
        '''
        print('----*Data Pre-processing*----')
        print('Normalizing expression data...')
        D1 = workflow.DESeq2(self.D0)
        print('D1: ',D1.shape)
        print('Averaging replicates...')
        D2 = workflow.average_replicates(D1)
        print('D2: ',D2.shape)
        print('Removing genes with low expression and low variance...')
        D3 = workflow.filter_low_expression_and_variance(D2)
        print('D3: ',D3.shape)
        print('Computing LFC of expression values...')
        L0 = workflow.expression_LFC(D3,genotypes= self.P_control.index.tolist())
        print('L0: ',L0.shape)
        print('Removing genes with low variance in LFC values...')
        L1 = workflow.filter_low_variance_LFC(L0,var_th=lfc_var_th)
        print('L1: ',L1.shape)
        print('Computing LFC of phenotypic traits...')
        Pl = self.P_stress.combine(self.P_control, lambda s,c: np.log2((s+1)/(c+1)))
        print('Pl: ',Pl.shape)
        
        print('Saving outputs (L1 and Pl)...')
        L1.to_csv('test/output/L1.csv',index=True,header=True,sep=',')
        Pl.to_csv('test/output/Pl.csv',index=True,header=True,sep=',')
        
        self.L1 = L1
        self.Pl = Pl
        print('Done')
        print('-------------------------------------------------------------')
        
        
    def scaleFreeFitIndex(k,nBreaks=30):
        '''
        Weighted degree distribution. Fits a linear model between 
        the logarithm of the probability that a node belongs to a given 
        strenght interval and the strength mean of the nodes within each 
        strenght interval.
        
        :param k: Node strength. The sum of weights of links connected to each 
            node.
        :type k: list[float]
        :param nBreaks: number of bins in strength histograms
        :type nBreaks: int
        
        :return: scale-free topology fit (R^2 of the adjusted linear model),
            slope of the fitted linear model, logarithm of the probability 
            that a node belongs to a given strenght interval, strength 
            mean of the nodes within each strenght interval, adjusted linear 
            model prediction.
        :rtype: tuple(float,float,np.ndarray[float],np.ndarray[float],
                       np.ndarray[float])
        '''
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
        
        return lm.score(log_dk, log_pdk), lm.coef_[0], log_dk, log_pdk,lm.predict(log_dk)
    
    
    
    def pickSoftTreshold(corx,
                         power_vect = [i for i in range(1,11)] + [i for i in range(12,21,2)],
                         nBreaks=30):
        '''
        Analysis of scale free topology for multiple beta powers that helps to
        pick an appropriate beta power for network construction.
        
        :param corx: weighted correlation matrix. Shape (n2,n2)
        :type corx: np.ndarray[float]
        :param power_vect: possible beta values to force a scale-free network
        :type power_vect: list[int]
        :param nBreaks: number of intervals to compute weighted network degree 
            distribution
        :type nBreaks: int
        
        :return: scale-free statistics for each power beta: beta value, R^2 
            of the weighted degree distribution, slope of the linear model 
            fitted to the degree distribution, network average node strength,
            network median node strength, network maximum node strength.
        :rtype: DataFrame[int,float,float,float,float,float]
        '''
        
        # node strenght of nodes
        datk = [[0 for _ in range(corx.shape[0])] for _ in range(len(power_vect))]
    
        # qué potencias de la matriz de coexpresión voy a necesitar
        power_steps = [power_vect[0]]
        for i in range(1,len(power_vect)):
            power_steps.append(power_vect[i]-power_vect[i-1])
        unique_power_steps = list(set(power_steps))
        
        # potencias de la matriz de coexpresión que voy a necesitar
        corx_powers = {}
        for i in range(len(unique_power_steps)):
            corx_powers[unique_power_steps[i]] = np.power(corx,unique_power_steps[i])
    
        # reutilizar potencias anteriores de la matriz de coexpresión para calcular las siguientes
        corx_prev = np.ones((corx.shape[0],corx.shape[1]))
        for i in range(len(power_vect)):
            corx_cur = np.multiply(corx_prev,corx_powers[power_steps[i]])
            datk[i] = [sum(x)-1 for x in zip(*corx_cur)]
            corx_prev = corx_cur
        
        # scale free index
        datout = pd.DataFrame(columns=['Power','SFT_Rsq','slope', 'kmean','kmedian','kmax'])
        datout['Power'] = power_vect
        
        for i in range(len(power_vect)):
            khelp = datk[i]
            SFT1 = workflow.scaleFreeFitIndex(k=khelp,nBreaks=nBreaks)
            
            datout.at[i,'SFT_Rsq'] = SFT1[0]
            datout.at[i,'slope'] = SFT1[1]
            datout.at[i,'kmean'] = np.mean(khelp)
            datout.at[i,'kmedian'] = np.median(khelp)
            datout.at[i,'kmax'] = max(khelp)
    
        return datout

    def coexpression_network(self,beta_th=0.8,plot=False):
        '''
        Build a co-expression network with a forced scale-free degree distribution.
        
        :param beta_th: desired minimum scale free topology fitting index (R^2)
        :type beta_th: float 
        :param plot: if set to TRUE a plot comparing the phenotypic traits
            under control and stress will be produced.
        :type plot: bool
        '''
        print('----*Co-expression Network Construction*----')
        # absolute pearson correlation between rows
        print('Computing absolute pearson correlation coefficient...')
        S = self.L1.transpose().corr(method='pearson').abs()
        print('S: ',S.shape)
        
        # force scale-free
        print('Testing beta values to get a scale-free network...')
        SF_tab = workflow.pickSoftTreshold(corx = S.values, power_vect=[1,2,3,4,5])
        beta = SF_tab.Power[SF_tab.SFT_Rsq > beta_th].min()
        if np.isnan(beta):
            beta = 1
        self.beta = beta
        print('beta = ',beta)
        A = S.pow(beta)
        print('A : ',A.shape)
        
        print('Saving scale-free adjacency matrix...')
        A.to_csv('test/output/A.csv',index=True,header=True,sep=',')
        self.A = A
        
        if plot:
            print('Creating plot of degree distribution for S and A...')
            plot_scale_free_adjustment(S,A,beta)
            plt.savefig('test/output/figures/scale_free.png',dpi=400, bbox_inches='tight', pad_inches=0.02)
        
        print('Done')
        print('-------------------------------------------------------------')
        
        
    def edge2coms_to_node2coms(edge_community_map,total_nodes):
        '''
        Convert a dictionary mapping edges to communities into a dictionary 
        mapping nodes to communities.
        
        :param edge_community_map: mapping of edges to the communities to which 
            they belong.
        :type edge_community_map: collections.defaultdict[tuple[int,int]->list[int]]
        
        :return: mapping of the nodes to the communities to which they belong.
        :rtype: dictionary[int->list[int]] 
        '''
        
        node2coms = dict()
        for k,v in edge_community_map.items():
            if k[0] in node2coms:
                node2coms[k[0]] = list(set(node2coms[k[0]] + v))
            else:
                node2coms[k[0]] = v
            if k[1] in node2coms:
                node2coms[k[1]] = list(set(node2coms[k[1]] + v))
            else:
                node2coms[k[1]] = v
        
        # complete unclustered nodes
        i = 0
        while len(node2coms) < total_nodes:
            if not i in node2coms:
                node2coms[i] = []
            i+=1        
        return node2coms
    
    def edge2coms_to_com2nodes(edge_community_map,total_nodes):
        '''
        Convert a dictionary mapping edges to communities into a dictionary 
        mapping communities to nodes.
        
        :param edge_community_map: mapping of edges to the communities to which 
            they belong.
        :type edge_community_map: collections.defaultdict[tuple[int,int]->list[int]]
        :param total_nodes: total number of nodes in the network.
        :type total_nodes: int
        
        :return: mapping of the communities to the member nodes
        :rtype: dictionary[int->list[int]] 
        '''
        com2nodes = dict()
        clustered_nodes = set()
        
        for (n1,n2),coms in edge_community_map.items():
            clustered_nodes = clustered_nodes.union([n1,n2])
            for c in coms:
                if c in com2nodes:
                    com2nodes[c] = list(set(com2nodes[c] + [n1,n2]))
                else:
                    com2nodes[c] = [n1,n2]
    
        unclustered_nodes = set(range(total_nodes)).difference(clustered_nodes)
        com2nodes[-1] = list(unclustered_nodes)        
        
        return com2nodes
        
    def affiliation_matrix(com2nodes,total_nodes,min_mod_size=3):
        '''
        Creates an affiliation matrix where rows are nodes and columns are 
        communities. Each entry in the matrix is equal to ​​1 if the node belongs
        to the community and is equal to 0 otherwise.
        
        :param com2nodes: mapping of communities to the member nodes
        :type com2nodes: dictionary[int->list[int]]
        :param total_nodes: total number of nodes in the network.
        :type total_nodes: int
        :param min_mod_size: minimal modules size
        :type min_mod_size: int
        
        :return: affiliation of nodes to zero or multiple communities
        :rtype: np.ndarray[bool]
        '''
    
        # remove the modules that do not have the minimal genes
        new_com2nodes = {key:val for key, val in com2nodes.items() if len(val) >= min_mod_size}
        new_com2nodes.pop(-1,None) # com -1 is the one with the unclustered nodes
        total_coms = len(new_com2nodes)
        
        F = np.zeros([total_nodes,total_coms])
        
        for c,nodes in new_com2nodes.items():
            if c != -1:
                for n in nodes:
                    F[n][c] = 1
        
        return F
    
    def module_identification(self,plot=False):
        '''
        Identify modules of overlapping communities from a co-expression 
        network using Hierarchical Link Clustering (HLC)
        
        :param plot: if set to True figures will be created summarizing 
            the overlap of the nodes.
        :type plot_hlc: bool
        '''
        
        print('----*Identification of co-expression modules*----')
        
        # unweighted network
        print('Computing unweighted network...')
        cutoff = 0.2
        A_hat = (self.A >= cutoff) * 1
        A_hat.to_csv('test/output/A_unweighted.csv',index=True,header=True,sep=',')
        
        G = nx.Graph(A_hat.values)
        G.remove_edges_from(nx.selfloop_edges(G)) # remove self loops
        print(nx.info(G))
        
        # HLC clustering
        print('Computing modules using HLC...')
        hlc_coms = algorithms.hierarchical_link_community(G)
        edge2com_dict = hlc_coms.to_edge_community_map()
        node2coms_dict = workflow.edge2coms_to_node2coms(edge2com_dict,
                                                         G.number_of_nodes())
        com2nodes_dict = workflow.edge2coms_to_com2nodes(edge2com_dict,
                                                         G.number_of_nodes())
        
        # build affiliation matrix
        self.F = workflow.affiliation_matrix(com2nodes_dict, G.number_of_nodes())
        Fs = sparse.csr_matrix(self.F)
        print('Saving affiliation matrix into file...')
        sparse.save_npz("test/output/F_hlc.npz", Fs)
            
        if plot:
            print('Creating overlapping module detection figures...')
            plot_overlapping_histogram(self.F)
            plt.savefig('test/output/figures/overlapping_histogram.png',
                        dpi=400, bbox_inches='tight', pad_inches=0.02)
            plot_overlapping_pie(self.F)
            plt.savefig('test/output/figures/overlapping_pie.png',
                        dpi=400, bbox_inches='tight', pad_inches=0.02)
    
    def aff_to_com2nodes(F):
        '''
        Uses the affiliation matrix to buil a dictionary mapping communities
        to nodes.
        
        :param F: affiliation matrix
        :type F: np.ndarray[bool]
        '''
        com2nodes = dict()
        for com in range(F.shape[1]):
            com2nodes[com] = [n for n in range(F.shape[0]) if F[n][com]]
        return com2nodes
    
    def module_eigengenes(L1t,modules_dict,min_mod_size = 3,node_names=None):
        '''
        Computes the module eigengenes as the first principal component of 
        the differential expression profiles for each community.
        
        :param L1t: LFC of the expression for notable genes. shape (m,n2)
        :type L1t: DataFrame
        :param modules_dict: dictionary where keys are modules and values are 
            list of genes
        :type modules_dict: dictionary[int->list[int]]
        :param min_mod_size: minimal modules size
        :type min_mod_size: int
        
        :return: dataframe where each column is the first Principal Component (PC)
            of the gene expression profiles of a given module. Shape (m,c)
        :rtype: DataFrame    
        '''
        if (node_names is None):
            node_names = np.array(L1t.columns.tolist())
        total_coms = len(modules_dict)
        moduleEigengen = pd.DataFrame(index = L1t.index)
        sub_module_dict = dict()
        for i in range(total_coms):
            if len(modules_dict[i]) >= min_mod_size:
                sub_module_dict[i] = modules_dict[i]                
                x = L1t.loc[:,node_names[modules_dict[i]]]
                moduleEigengen[i] = PCA(n_components=1).fit_transform(x)
        return moduleEigengen
    
    def lasso_module_selection_r2py(X,y,trait,plot=False):
        '''
        Performs a module selection using LASSO from glmnet library in R.
        LASSO adjust a regularized regresision model where the regresor 
        variables are the module eigenges and the outcome is a phenotypic trait.
        Regularization parameter is adjusted with cross-validation.
        LASSO reduces to zero the weights associated to modules with non-
        essential effects in the phenotypic response. Modules with non-zero
        coefficients are the selected ones, considered relevant to the 
        phenotypic response.
        
        :param X: Matrix of regresor variables, where each column corresponds
            to a module eigengene. Shape (m,c).
        :type X: np.ndarray[float]
        :param y: Output variable corresponding to a phenotypic trait. Shape (m,)
        :type y: np.ndarray[float]
        :param trait: phenotypic trait name or identifier
        :type trait: str
        '''
        R_lasso_string = '''
        R_lasso <- function(X,y,trait,rplot){
            library(glmnet)
            set.seed(1)
            y = as.vector(y)
            cv_lasso = cv.glmnet(X, y, alpha = 1)
            
            if (rplot == TRUE) {
                    png(filename=paste("test/output/figures/lassocv_",trait,".png"))
                    plot(cv_lasso,xlab="",ylab="",cex.axis=1.8)
                    title(ylab = 'Mean-Squared Error', line = 2.7, cex.lab = 1.8)
                    title(xlab = expression(paste('Log (', lambda, ") ")), line = 2.7, cex.lab = 1.8)
                    dev.off()
                    }
            
            model <- glmnet(X, y, alpha = 1, lambda = cv_lasso$lambda.min)
            bestlambda_model = coef(model)
            df = as.data.frame(summary(bestlambda_model))
            ans = as.vector(df[c(-1),c(1)])
            }
        '''
        r_pkg = STAP(R_lasso_string, "r_pkg")
        
        # convert python numpy matrices to R objects
        rX = numpy2ri(X)
        ry = numpy2ri(y)
        rplotcv = numpy2ri(np.array([plot]))
        
        
        # pass R objects into function
        py_ans = r_pkg.R_lasso(rX, ry, trait, rplotcv)
        
        return np.array(py_ans)-2
    
    def module_selection(self,plot=False):
        '''
        Each module is represented by an eigengene, which is associated with 
        each phenotypic trait using LASSO as a feature selection mechanism.
        
        :param plot: if set to true, figures will be created showing 
            cross-validation of LASSO for each phenotypic trait.
        :type plot: bool
        
        :return: Dictionary mapping the selected modules to their 
            corresponding genes
        :rtype: dictionary[str->list[str]]
        '''
        print('----*Detection of modules relevant to Phenotypic response*----')
        print('F: ',self.F.shape)            
        
        com2nodes_dict = workflow.aff_to_com2nodes(self.F)
        print('Computing eigengens...')
        M = workflow.module_eigengenes(self.L1.transpose(),com2nodes_dict)
        print('M: ',M.shape)
        print('Saving eigengene matrix (M)...')
        M.to_csv('test/output/M.csv',index=True,header=True,sep=',')
        self.M = M
        
        print('Module selection with LASSO...')
        node_names = np.array(self.L1.index.tolist())
        selected_mods = dict()
        I = set()
        for z in range(self.Pl.shape[1]):
            sm = workflow.lasso_module_selection_r2py(M.values,
                                                      self.Pl.iloc[:,z].values,
                                                      trait = self.Pl.columns[z],
                                                      plot=plot)
            sn = dict()
            for mod in sm:
                genes = list(node_names[com2nodes_dict[mod]])
                sn['mod_'+str(mod)] = genes
                I = I.union(genes)
            selected_mods[self.Pl.columns[z]] = sn
        
        self.I = I
        
        print('Done')
        print('-------------------------------------------------------------')
        
        return selected_mods
            
        
        
        
            
        
        
        
        
    
        
        
        
        
        
        
        
    
    
    
    
        
   
        
    
    
    
    

        
        