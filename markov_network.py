"""Implementation of the markov network simulator model."""

from functools import partial
import itertools

import elfi
import numpy as np
import scipy.stats as ss


def find_lexicographic_ordering(ug):
    """Lexicographic breadth-first search to find a lexicographic ordering of the nodes in an undirected graph (ug).

    Parameters
    ----------
    ug : float
        Adjacency matrix of undirected graph.

    Returns
    -------
    lex_ordering : 

    """
    d = ug.shape[0]
    lex_ordering = np.empty(shape=0)
    Sigma = np.arange(d)
    while Sigma.size > 0:
        tmp = np.atleast_1d(Sigma)
        v = np.atleast_1d(tmp[0])

        if len(tmp) == 1:
            Sigma = np.delete(Sigma, 0)
        else:
            tmp = tmp[1:]
            Sigma = tmp

        lex_ordering = np.append(lex_ordering, v)
        W = np.setdiff1d(np.where(ug[v, :] == 1)[0], lex_ordering)
        pos = 0
        while W.size != 0:
            T = np.intersect1d(Sigma[pos], W)
            if T.size != 0:
                U = np.setdiff1d(Sigma[pos], T)
                if U.size == 0:
                    Sigma[pos] = T
                else:
                    Sigma[pos] = U
                    Sigma = np.concatenate([Sigma[:pos], T, Sigma[pos:]])
                    pos += 1

                W = np.setdiff1d(W, T)
            pos += 1

    return lex_ordering.astype(int)


def find_perfect_elimination_ordering(ug):
    """Find perfect elimination ordering of nodes in an undirected graph (ug).

    Parameters
    ----------
    ug : float
        Adjacency matrix of undirected graph.

    Returns
    -------
    perf_elim_ordering :

    """
    lex_ordering = find_lexicographic_ordering(ug)
    for i in np.arange(2,lex_ordering.size):
        v = lex_ordering[i]
        v_nb = np.where(ug[v, :] == 1)[0]
        pos = np.where(np.in1d(lex_ordering[:i], v_nb))[0]
        if pos.size != 0:
            v_enb = lex_ordering[pos]
            w = np.atleast_1d(lex_ordering[pos[-1]])
            w_enb = np.setdiff1d(np.where(ug[w,:] == 1)[0],
                                 lex_ordering[pos[-1]:])
            tmp = np.setdiff1d(v_enb, np.concatenate((w, w_enb)))
            if tmp.size != 0:
                perf_elim_ordering = np.empty(0)
                raise ValueError('The graph is not chordal. There is no perfect elimination ordering!')
                return

    perf_elim_ordering = np.flip(lex_ordering)

    return perf_elim_ordering

    def ug_to_dag(ug):
    """Transforms chordal UG to equivalent DAG.

    Parameters
    ----------
    ug : float
        Adjacency matrix of undirected graph.

    Returns
    -------
    dag : 
    node_ordering : 
    """

    # Find perfect elimination ordering of ug
    perf_elim_ordering = find_perfect_elimination_ordering(ug)

    # If an perfect elimination ordering exists, construct a DAG according to it
    if perf_elim_ordering.size == 0:
        dag = np.empty(0)
    else:
        dag = np.zeros(ug.shape)
        for i in np.arange(len(perf_elim_ordering)):
            node = perf_elim_ordering[i]
            nb = np.where(ug[node, :] == 1)[0]
            par = np.setdiff1d(nb, perf_elim_ordering[:i])
            dag[par, node] = 1

    node_ordering = np.flip(perf_elim_ordering)

    return dag, node_ordering


def bn_sample_joint_from_prior(dag, node_ordering, n, ess, random_state=None):
    """Sample joint distributions from a BDeu prior for a given DAG. 

    Parameters
    ----------
    dag : float or arraylike
        adjacency matrix of directed acyclic graph.
    node_ordering : float
        node ordering in which the nodes are samples. 
    n : int
        number of samples.
    ess : float
        effective sample size in BDeu prior.

    Returns
    -------
    oc : joint outcomes
    pmat : matrix of sampled joint distributions.


    """
    d = dag.shape[1]

    repeated_list = [[True, False]] * d
    oc = np.array(list(itertools.product(*repeated_list)))
    pmat = np.ones((oc.shape[0],n))
    for i in np.arange(d):
        node = node_ordering[i]
        par = np.where(dag[:,node] == 1)[0]
        if par.size != 0:
            repeated_list = [[True, False]] * np.sum(par)
            par_oc = np.array(list(itertools.product(*repeated_list)))
            alpha = ess / (oc.shape[0] * 2)
            for l in np.arange(par_oc.shape[0]):
                p0 = np.atleast_1d(ss.beta.rvs(alpha, alpha, size=n))
                ind = np.equal(oc[:,par], par_oc[l,:]).all(axis=1)
                ind0 = np.logical_and(ind.ravel(), (oc[:,node].ravel() == 0))
                pmat[ind0,:] = pmat[ind0,:] * np.tile(p0, (np.sum(ind0), 1))
                ind1 = np.logical_and(ind.ravel(),(oc[:,node] == 1).ravel())
                pmat[ind1,:] = pmat[ind1,:] * np.tile(1 - p0, (np.sum(ind1), 1))
        else:
            alpha = ess / 2
            p0 = np.atleast_2d(ss.beta.rvs(alpha, alpha, size=n))
            ind0 = (oc[:,node] == 0)
            pmat[ind0,:] = pmat[ind0,:] * np.tile(p0, (np.sum(ind0), 1))
            ind1 = (oc[:,node] == 1)
            pmat[ind1,:] = pmat[ind1,:] * np.tile(1 - p0, (np.sum(ind0), 1))

    return oc, pmat


def maximalCliques(A):
    """Find maximal cliques using the Bron-Kerbosch algorithm
    %MAXIMALCLIQUES 
    %   Given a graph's boolean adjacency matrix, A, find all maximal cliques 
    %   on A using the Bron-Kerbosch algorithm in a recursive manner.  The 
    %   graph is required to be undirected and must contain no self-edges.
    %
    %   V_STR is an optional input string with the version of the Bron-Kerbosch 
    %   algorithm to be used (either 'v1' or 'v2').  Version 2 is faster (and 
    %   default), and version 1 is included for posterity.
    %
    %   MC is the output matrix that contains the maximal cliques in its 
    %   columns.
    %
    %   Note: This function can be used to compute the maximal independent sets
    %   of a graph A by providing the complement of A as the input graph.  
    %
    %   Note: This function can be used to compute the maximal matchings of a 
    %   graph A by providing the complement of the line graph of A as the input
    %   graph.
    %
    %   Ref: Bron, Coen and Kerbosch, Joep, "Algorithm 457: finding all cliques
    %   of an undirected graph", Communications of the ACM, vol. 16, no. 9, 
    %   pp: 575–577, September 1973.
    %
    %   Ref: Cazals, F. and Karande, C., "A note on the problem of reporting 
    %   maximal cliques", Theoretical Computer Science (Elsevier), vol. 407,
    %   no. 1-3, pp: 564-568, November 2008.
    %
    %   Jeffrey Wildman (c) 2011
    %   jeffrey.wildman@gmail.com
    %   
    %   Updated: 10/27/2011 - updated documentation & removal of ~ punctuation 
    %   to ignore function output arguments for better compatibility with older
    %   MATLAB versions prior to 2009b (Thanks to Akli Benali).
    % first, some input checking
    if size(A,1) ~= size(A,2)
        error('MATLAB:maximalCliques', 'Adjacency matrix is not square.');
    elseif ~all(all((A==1) | (A==0)))
        error('MATLAB:maximalCliques', 'Adjacency matrix is not boolean (zero-one valued).')
    elseif ~all(all(A==A.'))
        error('MATLAB:maximalCliques', 'Adjacency matrix is not undirected (symmetric).')
    elseif trace(abs(A)) ~= 0
        error('MATLAB:maximalCliques', 'Adjacency matrix contains self-edges (check your diagonal).');
    end
        
    if ~exist('v_str','var')
        v_str = 'v2';
    end
    if ~strcmp(v_str,'v1') && ~strcmp(v_str,'v2')
        warning('MATLAB:maximalCliques', 'Version not recognized, defaulting to v2.');
        v_str = 'v2';
    end               
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Adjacency matrix is not square.')
    elif np.trace(A) != 0:
        raise ValueError('Adjacency matrix contains self-edges.')
    elif ~(A == A.T).all():
        raise ValueError('Adjacency matrix is not symmetric.')
    elif ~(np.logical_or(A == 0, A == 1).all()):
        raise ValueError('Adjacency matrix is not boolean.')

    n = A.shape[1]               # number of vertices
    MC = np.empty(shape=[n, 0])  # storage for maximal cliques
    R = np.empty(shape=[1, 0])   # currently growing clique
    P = np.arange(n)             # prospective nodes connected to all nodes in R
    X = np.empty(shape=[1, 0])   # nodes already processed

    def BKv2( R, P, X, MC, A, n ):

        if P.size == 0 and X.size == 0:
            # report R as a maximal clique
            # print("P.size == 0 and X.size == 0")

            newMC = np.zeros((1, n))
            newMC[0,R.astype(int)] = 1               # newMC contains ones at indices equal to the values in R   
            MC = np.concatenate([MC, newMC.T], axis=1)
        else:
            # choose pivot
            ppivots = np.union1d(P, X)           # potential pivots
            binP = np.zeros((1, n))
            binP[0,P] = 1                    # binP contains ones at indices equal to the values in P          
            # rows of A(ppivots,:) contain ones at the neighbors of ppivots
            pcounts = np.dot(A[ppivots.astype(int), :], binP.T)   # cardinalities of the sets of neighbors of each ppivots intersected with P
            
            max_pos = np.argmax(pcounts)

            u_p = ppivots[max_pos]             # select one of the ppivots with the largest count

            for u in np.intersect1d(np.where(A[u_p.astype(int), :] == 0)[0], P):  #% all prospective nodes who are not neighbors of the pivot
                P = np.setdiff1d(np.union1d(P, u), np.intersect1d(P, u))
                Rnew = np.concatenate([R, np.atleast_2d(u)], axis=1)
                Nu = np.where((A[u.astype(int), :] > 0))[0]
                Pnew = np.intersect1d(P, Nu)
                Xnew = np.atleast_2d(np.intersect1d(X, Nu))
                MC = BKv2(Rnew, Pnew, Xnew, MC, A, n)
                X = np.concatenate([X, np.atleast_2d(u)], axis=1)

        return MC
    MC = BKv2(R, P, X, MC, A, n)

    return MC


def mn_sample_para(ug, n, ess):
    '''
    Sample model parameters of a chordal MN from a prior/posterior specified .
    Parameters
    ----------
    ug : float
        Adjacency matrix of the undirected (must be chordal).
    n : int
        Number of samples
    ess : float
        Effective sample size in the BDeu prior. 

    Returns
    -------
    para_mat - MN parameters.
    para_sample - matrix containing the n samples
    # oc - joint outcome space.
    # pmat - matrix of sampled joint distributions.
    '''


    # Turn DAG into an equivalent UG
    dag, node_ordering = ug_to_dag(ug)

    # IF no data, sample joint distributions from the prior, ELSE
    # sample joint distributions from posterior.
    oc, pmat = bn_sample_joint_from_prior(dag, node_ordering, n, ess)

    # Convert the joint distributions into MN parameters.
    para_mat, para_sample = joint_to_mn_para(oc,pmat,ug)

    return para_mat, para_sample


def joint_to_mn_para(oc,pmat,ug):
    """Generates the log-linear parameterization of a collection of Markov network distributions.
    
    function [para_mat,para_sample] = joint_to_mn_para(oc,pmat,ug)
    %-------------------------------------------------------------------------
    % Generates the log-linear parameterization of a collection of Markov 
    % network distributions.  
    % INPUT:    oc - joint outcome space.
    %           pmat - matrix of joint distributions (col vector if only one)
    %           that stisfies the independence assumpitions implied by ug.
    %           ug - (symmetric) adjacency matrix of a undirected graph.
    % OUTPUT:   para_mat - matrix representing the log-linear parameters.
    %           para_sample - parameter values, same number of rows as
    %           para_mat and same number of columns as pmat.
    %-------------------------------------------------------------------------
    
    """
    para_mat = mn_para_mat(ug)
    para_sample = np.zeros((para_mat.shape[0], pmat.shape[1]))
    para_size = np.sum(para_mat, axis=1)    
    for k in np.unique(para_size):
        ind = np.where(para_size == k)[0]
        for i in ind:
            para = para_mat[i,:]
            ind0 = np.equal(oc, para).all(axis=1)
            p = pmat[ind0,:]
            if k == 0:
                para_sample[i,:] = np.log(p)
            else:
                pos = ~np.any(para_mat[:, ~para], axis=1)
                pos[i] = False
                para_sample[i, :] = np.log(p) - np.sum(para_sample[pos,:], axis=0)

    return para_mat, para_sample

def mn_para_mat(ug):
    """Generates the log-linear parameterization of an undirected graph (UG) 
    function para_mat = mn_para_mat(ug)
    %-------------------------------------------------------------------------
    % 
    % INPUT: ug - (symmetric) adjacency matrix of a chordal undirected graph
    % OUTPUT: para_mat - matrix representing the log-linear parameters 
    %-------------------------------------------------------------------------
    end
    """

    d = ug.shape[1]
    mc = maximalCliques(ug)
    mc_size = np.sum(mc,0).astype(int)
    m = (2 ** mc_size).astype(int)
    
    # para_mat = false(sum(m),d)
    para_mat = np.full((np.sum(m), d), False, dtype=bool)
    pos = 0
    for clq in np.arange(mc.shape[1]):
        # repeated_list = [[False, True]] * d
        # 
        repeated_list = [[False, True]] * mc_size[clq]
        oc = np.array(list(itertools.product(*repeated_list)))
        # tmp = repmat({[false true]},1,mc_size(clq));
        para_mat[pos:(pos + m[clq]), mc[:,clq] == 1] = oc
        pos = pos + m[clq]

    para_mat = np.unique(para_mat, axis=0)
    # para_mat = unique(para_mat, 'rows');

    return para_mat


def mn_generate_data(para_mat, para_val, n, seed=None):
    '''Generates data from a log-linear model using exact sampling.

    Parameters
    ----------
    para_mat : float or arraylike
        Logical matrix representing log-linear parameters.
    para_val : float or arraylike
        Numerical values of the log-linear parameters.
    n : int
        number of samples.

    Returns
    -------
    data : Sampled data matrix.


    '''
    oc, p = mn_joint_dist(para_mat, para_val)
    ind = np.repeat(np.arange(oc.shape[0]), ss.multinomial(n, p, seed=seed))
    data = oc[ind, :]

    return data


def mn_joint_dist(para_mat, par_val):
    ''' Calculate joint distribution from a log-linear parameterization.

    Parameters
    ----------
    para_mat : float or arraylike
        Logical matrix representing log-linear parameters.
    para_val : float or arraylike
        Numerical values of the log-linear parameters.

    Returns
    -------
    oc : joint outcomes of the variables.
    p : probabilities of the joint outcomes.

    '''
    d = para_mat.shape[1]
    repeated_list = [[True, False]] * d
    oc = np.array(list(itertools.product(*repeated_list)))
    p = np.zeros(oc.shape[0])
    for i in np.arange(para_mat.shape[0]):
        ind = np.all(oc[:, para_mat[i, :]], axis=1)
        p[ind] = p[ind]+para_val[i]
    end
    p = np.exp(p)
    if np.abs(np.sum(p)-1) > 1e-6:
        ValueError('Check your model: abs(sum(p)-1) > 1e-6.')
    else:
        p = p / np.sum(p)
    end

    return oc, p
