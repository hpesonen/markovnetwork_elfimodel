"""Implementation of the markov network simulator model."""

from functools import partial
import itertools

import elfi
import numpy as np
import scipy.stats as ss


def find_lexicographic_ordering(ug):
    """Lexicographic breadth-first search to find a.

        lexicographic ordering of the nodes in an undirected graph (ug).

    Parameters
    ----------
    ug : boolean array
        Adjacency matrix of undirected graph.

    Returns
    -------
    lex_ordering : integer array
        Lexicographic ordering of the nodes.

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
    ug : boolean matrix
        Adjacency matrix of undirected graph.

    Returns
    -------
    perf_elim_ordering : integer array
        Perfect elimination ordering of the nodes.

    """
    lex_ordering = find_lexicographic_ordering(ug)
    for i in np.arange(2, lex_ordering.size):
        v = lex_ordering[i]
        v_nb = np.where(ug[v, :] == 1)[0]
        pos = np.where(np.in1d(lex_ordering[:i], v_nb))[0]
        if pos.size != 0:
            v_enb = lex_ordering[pos]
            w = np.atleast_1d(lex_ordering[pos[-1]])
            w_enb = np.setdiff1d(np.where(ug[w, :] == 1)[0],
                                 lex_ordering[pos[-1]:])
            tmp = np.setdiff1d(v_enb, np.concatenate((w, w_enb)))
            if tmp.size != 0:
                perf_elim_ordering = np.empty(0)
                raise ValueError('The graph is not chordal. '
                                 'There is no perfect elimination ordering!')
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

    # If an perfect elimination ordering exists,
    # construct a DAG according to it.
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

    oc = get_joint_outcomes(dag.shape[1])

    return dag, node_ordering, oc


def get_joint_outcomes(d):
    """Get the matrix of joint outcomes for the dag."""
    repeated_list = [[True, False]] * d
    oc = np.array(list(itertools.product(*repeated_list)))

    return oc


def bn_sample_joint_from_prior(beta, dag, node_ordering, n, ess, batch_size=1, random_state=None):
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

    pmat = np.ones((oc.shape[0], beta.shape[0]))
    k = 0
    for i in np.arange(d):
        node = node_ordering[i]
        par = np.where(dag[:, node] == 1)[0]

        if par.size != 0:
            repeated_list = [[True, False]] * (par.size)
            par_oc = np.array(list(itertools.product(*repeated_list)))
            alpha = ess / (oc.shape[0] * 2)

            for l in np.arange(par_oc.shape[0]):
                p0 = beta[:, k]
                k += 1
                ind = np.equal(oc[:, par], par_oc[l, :]).all(axis=1)

                ind0 = np.logical_and(ind.ravel(), (oc[:, node] == 0).ravel())
                pmat[ind0, :] = pmat[ind0, :] * np.tile(p0.reshape(1, -1),
                                                        (np.sum(ind0), 1))

                ind1 = np.logical_and(ind.ravel(), (oc[:, node] == 1).ravel())
                pmat[ind1, :] = pmat[ind1, :] * np.tile(1 - p0.reshape(1, -1),
                                                        (np.sum(ind1), 1))
        else:
            alpha = ess / 2
            p0 = beta[:, k]
            k += 1

            ind0 = (oc[:, node] == 0)
            pmat[ind0, :] = pmat[ind0, :] * np.tile(p0.reshape(1, -1),
                                                    (np.sum(ind0), 1))

            ind1 = (oc[:, node] == 1)
            pmat[ind1, :] = pmat[ind1, :] * np.tile(1 - p0.reshape(1, -1),
                                                    (np.sum(ind0), 1))

    return pmat


def maximalCliques(A):
    """Find maximal cliques using the Bron-Kerbosch algorithm given a graph's
       boolean adjacency matrix.

       Ref: Bron, Coen and Kerbosch, Joep, "Algorithm 457: finding all cliques
       of an undirected graph", Communications of the ACM, vol. 16, no. 9,
       pp: 575–577, September 1973.

       Ref: Cazals, F. and Karande, C., "A note on the problem of reporting
       maximal cliques", Theoretical Computer Science (Elsevier), vol. 407,
       no. 1-3, pp: 564-568, November 2008.

       Adapted from a matlab-function by Jeffrey Wildman (c) 2011
       jeffrey.wildman@gmail.com

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

    def BKv2(R, P, X, MC, A, n):

        if P.size == 0 and X.size == 0:
            # report R as a maximal clique
            newMC = np.zeros((1, n))
            # newMC contains ones at indices equal to the values in R
            newMC[0, R.astype(int)] = 1
            MC = np.concatenate([MC, newMC.T], axis=1)
        else:
            # choose pivot
            # potential pivots
            ppivots = np.union1d(P, X)
            binP = np.zeros((1, n))
            # binP contains ones at indices equal to the values in P
            binP[0, P] = 1
            # Rows of A(ppivots,:) contain ones at the neighbors of ppivots.
            # Cardinalities of the sets of neighbors of each ppivots 
            # intersected with P.
            pcounts = np.dot(A[ppivots.astype(int), :], binP.T)
            max_pos = np.argmax(pcounts)
            # Select one of the ppivots with the largest count.
            u_p = ppivots[max_pos]
            # All prospective nodes who are not neighbors of the pivot
            for u in np.intersect1d(
                     np.where(A[u_p.astype(int), :] == 0)[0], P
                     ):
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


def mn_sample_para(beta, ug, ess, dag, node_ordering, oc, para_mat, batch_size=1):
    """Sample model parameters of a chordal MN from a prior/posterior specified.

    Parameters
    ----------
    ug : boolean array
        Adjacency matrix of the undirected (must be chordal).
    n : int
        Number of samples
    ess : float
        Effective sample size in the BDeu prior.

    Returns
    -------
    para_mat - MN parameters.
    para_sample - matrix containing the n samples

    """

    n = 1
    pmat = bn_sample_joint_from_prior(beta=beta, dag=dag,
                                      node_ordering=node_ordering,
                                      n=n, ess=ess, batch_size=batch_size)

    # Convert the joint distributions into MN parameters.
    para_sample = joint_to_mn_para(oc=oc, pmat=pmat, ug=ug, para_mat=para_mat)

    return para_sample


def joint_to_mn_para(oc, pmat, ug, para_mat):
    """Generate the log-linear parameterization of a collection
       of Markov network distributions.

    Parameters
    ----------
    oc: 
        Joint outcome space.
    pmat:
        Matrix of joint distributions (col vector if only one).
    ug: boolean array
        Symmetric adjacency matrix of an undirected graph.

    Returns
    -------
    para_mat:
        Matrix representing the log-linear parameters.
    para_sample:
        Parameter values.

    """
    para_sample = np.zeros((para_mat.shape[0], pmat.shape[1]))
    para_size = np.sum(para_mat, axis=1)
    for k in np.unique(para_size):
        ind = np.where(para_size == k)[0]
        for i in ind:
            para = para_mat[i, :]
            ind0 = np.equal(oc, para).all(axis=1)
            p = pmat[ind0, :]
            if k == 0:
                para_sample[i, :] = np.log(p)
            else:
                pos = ~np.any(para_mat[:, ~para], axis=1)
                pos[i] = False
                para_sample[i, :] = np.log(p) - np.sum(para_sample[pos, :],
                                                       axis=0)

    return para_sample


def mn_para_mat(ug):
    '''Generates the log-linear parameterization of an undirected graph (UG).

    Parameters
    ----------
    ug: boolean array
        Symmetric adjacency matrix of a chordal undirected graph.

    Returns
    -------
    para_mat:
        Matrix representing the log-linear parameters.

    '''
    d = ug.shape[1]
    mc = maximalCliques(ug)
    mc_size = np.sum(mc, 0).astype(int)
    m = (2 ** mc_size).astype(int)

    para_mat = np.full((np.sum(m), d), False, dtype=bool)
    pos = 0
    for clq in np.arange(mc.shape[1]):
        repeated_list = [[False, True]] * mc_size[clq]
        oc = np.array(list(itertools.product(*repeated_list)))
        para_mat[pos:(pos + m[clq]), mc[:, clq] == 1] = oc
        pos = pos + m[clq]

    para_mat = np.unique(para_mat, axis=0)

    return para_mat


def gmn_simulate(*a, ug, n, ess, dag, node_ordering, oc, para_mat,
                 batch_size=1, random_state=None):
    beta = np.asanyarray(a).reshape(batch_size, -1)

    para_val = mn_sample_para(beta=beta, ug=ug, ess=ess, dag=dag,
                              node_ordering=node_ordering,
                              oc=oc, para_mat=para_mat)

    # Generate observed data
    obs_data_ind = np.empty((batch_size, n))
    for i in np.arange(batch_size):
        obs_data_ind[i, :] = mn_generate_data(para_mat=para_mat,
                                              para_val=para_val[:, i],
                                              n=n, oc=oc,
                                              batch_size=1,
                                              random_state=random_state)

    return obs_data_ind


def mn_generate_data(para_mat, para_val, n, oc, batch_size=1,
                     random_state=None):
    """Generate data from a log-linear model using exact sampling.

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


    """
    p = mn_joint_dist(para_mat, para_val, oc)

    # print(ss.multinomial.rvs(n, p, size=1, random_state=seed))
    ind = np.repeat(np.arange(oc.shape[0]),
                    ss.multinomial.rvs(n, p, size=1,
                    random_state=random_state).ravel())
    # data = oc[ind, :]

    return ind


def mn_joint_dist(para_mat, para_val, oc):
    """Calculate joint distribution from a log-linear parameterization.

    Parameters
    ----------
    para_mat : float or arraylike
        Logical matrix representing log-linear parameters.
    para_val : float or arraylike
        Numerical values of the log-linear parameters.
    oc : 
        Joint outcomes of the variables.

    Returns
    -------

    p : probabilities of the joint outcomes.

    """
    p = np.zeros(oc.shape[0])
    for i in np.arange(para_mat.shape[0]):
        ind = np.all(oc[:, para_mat[i, :]], axis=1)
        p[ind] = p[ind]+para_val[i]

    p = np.exp(p)
    if np.abs(np.sum(p)-1) > 1e-6:
        ValueError('Check your model: abs(sum(p)-1) > 1e-6.')
    else:
        p = p / np.sum(p)

    return p


def get_model(n_obs=100, ess=50, ug=None, seed_obs=None):
    """Return a network model for elfi.

    Parameters
    ----------
    n_obs : int, optional
        observation length of the MA2 process
    ess : in, optional
        effective sample size
    ug : boolean array
        undirected adjacency matrix
    seed_obs : int, optional
        seed for the observed data generation

    Returns
    -------
    m : elfi.ElfiModel

    """
    if ug is None:
        ug = np.zeros((4, 4))
        ug[0, 1:4] = 1
        ug[1:4, 0] = 1

    m = elfi.new_model()
    priors = []
    dag, node_ordering, oc = ug_to_dag(ug)
    para_mat = mn_para_mat(ug)
    combs_to_node = 2 ** np.sum(dag, axis=0)
    n_dim = np.sum(combs_to_node).astype(int)
    alpha = ess / 2 / oc.shape[0] * np.ones(n_dim)
    no_connections = np.where(np.sum(dag, axis=0) == 0)[0].astype(int)
    alpha[no_connections] = ess / 2

    for i in np.arange(n_dim):
        name_prior = 'a_{}'.format(i)
        prior_beta = elfi.Prior('beta',
                                alpha[i],
                                alpha[i],
                                model=m,
                                name=name_prior)
        priors.append(prior_beta)

    sim_fn = partial(gmn_simulate,
                     ug=ug,
                     n=n_obs,
                     ess=ess,
                     dag=dag,
                     node_ordering=node_ordering,
                     oc=oc,
                     para_mat=para_mat)
    a_true = 0.2 * np.ones((n_dim, 1))
    y = sim_fn(a_true)

    elfi.Simulator(sim_fn, *priors, observed=y, name='GMN')
    elfi.Summary(sumstats, m['GMN'], oc.shape[0], n_obs, name='S')
    elfi.Distance('euclidean',  m['S'], name='d')

    return m


def sumstats(x, n, n_obs):

    e = np.atleast_2d(np.arange(n))
    x = np.asanyarray(x).reshape(-1, n_obs)
    m = x.shape[0]
    if m > 1:
        tmp, counts = np.apply_along_axis(np.unique, 1, np.column_stack((x, e[np.zeros(m).astype(int), :])), return_counts=True)
    else:    
        tmp, counts = np.unique(np.column_stack((x, e[np.zeros(m).astype(int), :])), axis=1, return_counts=True)

    return counts - 1


def mn_calculate_jsd(*sim_data, observed, para_mat):
    '''function jsd = mn_calculate_jsd(para_mat,obs_data,sim_data)

    Parameters
    ----------
    sim_data : 
        Simulated dataset
    para_mat :
        Logical matrix representing log-linear parameters.
    observed :
        Observed dataset.

    Returns
    -------
    dist : 
        Jensen-Shannon distance between simulated and observed datasets

    '''
    if not np.array_equal(observed.shape, sim_data.shape):
        raise ValueError('Observed data and simulated data '
                         'are of different sizes.')

    inc = para_mat.any(axis=1)
    para_mat = para_mat[inc, :]
    jsd_vec = np.zeros(para_mat.shape[0])
    for i in np.arange(para_mat.shape[0]):
        ind = para_mat[i, :]
        n_ind = np.sum(para_mat[i, :])
        repeated_list = [[True, False]] * n_ind
        oc = np.array(list(itertools.product(*repeated_list)))
        noc = oc.shape[0]
        count_obs = np.zeros(noc)
        count_sim = np.zeros(noc)
        for j in np.arange(noc):
            count_obs[j] = np.sum(np.equal(observed[:, ind],
                                  oc[j, :]).all(axis=1)) + 1
            count_sim[j] = np.sum(np.equal(sim_data[:, ind],
                                  oc[j, :]).all(axis=1)) + 1
        p = count_obs / np.sum(count_obs)
        q = count_sim / np.sum(count_sim)
        m = (p + q) / 2
        jsd_vec[i] = 0.5 * (np.dot(p, np.log(p / m))
                            + np.dot(q, np.log(q / m)))

    jsd = np.mean(jsd_vec)

    return jsd
