import numpy as np
from scipy.special import logsumexp
import scipy
from EM_HYDRA.sinkornknopp import *
from sklearn.cluster import KMeans
import cvxpy as cp
from sklearn.mixture import GaussianMixture


def one_hot_encode(y, n_classes=None):
    ''' utils function in order to turn a label vector into a one hot encoded matrix '''
    if n_classes is None:
        n_classes = np.max(y) + 1
    y_one_hot = np.copy(y)
    return np.eye(n_classes)[y_one_hot]


def sigmoid(x, lambda_=5):
    return 1 / (1 + np.exp(-lambda_ * x))


def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def proportional_assign(l, d):
    """
    Proportional assignment based on margin
    :param l: int
    :param d: int
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    invL = np.divide(1, l)
    idx = np.isinf(invL)
    invL[idx] = d[idx]

    for i in range(l.shape[0]):
        pos = np.where(invL[i, :] > 0)[0]
        neg = np.where(invL[i, :] < 0)[0]
        if pos.size != 0:
            invL[i, neg] = 0
        else:
            invL[i, :] = np.divide(invL[i, :], np.amin(invL[i, :]))
            invL[i, invL[i, :] < 1] = 0

    S = np.multiply(invL, np.divide(1, np.sum(invL, axis=1))[:, np.newaxis])

    return S


def sample_dpp(evalue, evector, k=None):
    """
    sample a set Y from a dpp.  evalue, evector are a decomposed kernel, and k is (optionally) the size of the set to return
    :param evalue: eigenvalue
    :param evector: normalized eigenvector
    :param k: number of cluster
    :return:
    """
    if k == None:
        # choose eigenvectors randomly
        evalue = np.divide(evalue, (1 + evalue))
        evector = np.where(np.random.random(evalue.shape[0]) <= evalue)[0]
    else:
        v = sample_k(evalue, k)  # v here is a 1d array with size: k

    k = v.shape[0]
    v = v.astype(int)
    v = [i - 1 for i in
         v.tolist()]  # due to the index difference between matlab & python, here, the element of v is for matlab
    V = evector[:, v]

    ## iterate
    y = np.zeros(k)
    for i in range(k, 0, -1):
        # compute probabilities for each item
        P = np.sum(np.square(V), axis=1)
        P = P / np.sum(P)

        # choose a new item to include
        y[i - 1] = np.where(np.random.rand(1) < np.cumsum(P))[0][0]
        y = y.astype(int)

        # choose a vector to eliminate
        j = np.where(V[y[i - 1], :])[0][0]
        Vj = V[:, j]
        V = np.delete(V, j, 1)

        # Update V
        if V.size == 0:
            pass
        else:
            V = np.subtract(V, np.multiply(Vj, (V[y[i - 1], :] / Vj[y[i - 1]])[:,
                                               np.newaxis]).transpose())  # watch out the dimension here

        ## orthogonalize
        for m in range(i - 1):
            for n in range(m):
                V[:, m] = np.subtract(V[:, m], np.matmul(V[:, m].transpose(), V[:, n]) * V[:, n])

            V[:, m] = V[:, m] / np.linalg.norm(V[:, m])

    y = np.sort(y)

    return y


def sample_k(lambda_value, k):
    """
    Pick k lambdas according to p(S) \propto prod(lambda \in S)
    :param lambda_value: the corresponding eigenvalues
    :param k: the number of clusters
    :return:
    """

    # compute elementary symmetric polynomials
    E = elem_sym_poly(lambda_value, k)

    # iterate over the lambda value
    num = lambda_value.shape[0]
    remaining = k
    S = np.zeros(k)
    while remaining > 0:
        # compute marginal of num given that we choose remaining values from 0:num-1
        if num == remaining:
            marg = 1
        else:
            marg = lambda_value[num - 1] * E[remaining - 1, num - 1] / E[remaining, num]

        # sample marginal
        if np.random.rand(1) < marg:
            S[remaining - 1] = num
            remaining = remaining - 1
        num = num - 1
    return S


def elem_sym_poly(lambda_value, k):
    """
    given a vector of lambdas and a maximum size k, determine the value of
    the elementary symmetric polynomials:
    E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    :param lambda_value: the corresponding eigenvalues
    :param k: number of clusters
    :return:
    """
    N = lambda_value.shape[0]
    E = np.zeros((k + 1, N + 1))
    E[0, :] = 1

    for i in range(1, k + 1):
        for j in range(1, N + 1):
            E[i, j] = E[i, j - 1] + lambda_value[j - 1] * E[i - 1, j - 1]

    return E


def consensus_clustering(clustering_results, n_clusters, index_positives, negative_weighting='all',
                         cluster_weight=None):
    S = np.ones((clustering_results.shape[0], n_clusters)) / n_clusters
    co_occurrence_matrix = np.zeros((clustering_results.shape[0], clustering_results.shape[0]))

    for i in range(clustering_results.shape[0] - 1):
        for j in range(i + 1, clustering_results.shape[0]):
            if cluster_weight is None:
                co_occurrence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])
            else:
                co_occurrence_matrix[i, j] = sum(
                    cluster_weight * (clustering_results[i, :] == clustering_results[j, :]).astype(np.int))

    co_occurrence_matrix = np.add(co_occurrence_matrix, co_occurrence_matrix.transpose())
    # here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(co_occurrence_matrix, axis=1)), co_occurrence_matrix)

    Laplacian_norm = np.subtract(np.eye(clustering_results.shape[0]), np.matmul(
        np.matmul(np.diag(1 / np.sqrt(np.sum(co_occurrence_matrix, axis=1))), co_occurrence_matrix),
        np.diag(1 / np.sqrt(np.sum(co_occurrence_matrix, axis=1)))))
    # replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    # check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    e_value, e_vector = scipy.linalg.eigh(Laplacian_norm)

    # check if the eigen vector is complex
    if np.any(np.iscomplex(e_vector)):
        e_value, e_vector = scipy.linalg.eigh(Laplacian)

    # train Spectral Clustering algorithm and make predictions
    spectral_features = e_vector.real[:, 0:n_clusters]

    # apply clustering method according to negative weighting
    if negative_weighting in ['all']:
        k_means = KMeans(n_clusters=n_clusters).fit(spectral_features[index_positives])
        S[index_positives] = one_hot_encode(k_means.labels_.astype(np.int), n_classes=n_clusters)
    else : # negative_weighting in ['soft_clustering', 'hard_clustering']:
        gaussian_mixture = GaussianMixture(n_components=n_clusters).fit(spectral_features)
        S = gaussian_mixture.predict_proba(spectral_features)
        S = cpu_sk(S)
    '''
    elif negative_weighting in ['hard_clustering']:
        k_means = KMeans(n_clusters=n_clusters).fit(spectral_features)
        S = one_hot_encode(k_means.labels_.astype(np.int), n_classes=n_clusters)'''

    return S


def optimize_HYDRA_dual(self, X, y_polytope, S):
    # first we define number of samples, number of clusters etc...
    n_samples = X.shape[0]
    n_clusters = S.shape[1]
    diag_y = np.eye(n_samples, n_samples) * y_polytope[:, None]
    y_polytope_repeat = np.repeat(y_polytope[:, None], S.shape[1], axis=1)

    # then we define the Variables and Parameters
    lambda_dual_matrix = cp.Variable(shape=S.shape, nonneg=True)
    S_parameter = cp.Parameter(shape=S.shape, value=S, nonneg=True)
    y_polytope_parameter = cp.Parameter(shape=y_polytope_repeat.shape, value=y_polytope_repeat)
    K = diag_y @ X @ X.T @ diag_y
    K_parameter = cp.Parameter(shape=K.shape, PSD=True, value=K)

    # we define the objective function
    obj = - cp.sum(lambda_dual_matrix)
    for k in range(n_clusters):
        lambda_column = lambda_dual_matrix[:, k][:, None]
        obj += cp.quad_form(lambda_column, K_parameter)

    # We set the constraints
    const = [cp.multiply(y_polytope_parameter, lambda_dual_matrix) >= np.zeros((n_samples, n_clusters)),
             lambda_dual_matrix >= np.zeros(lambda_dual_matrix.shape),
             self.C * S_parameter >= lambda_dual_matrix]

    # we run the problem minimizer
    prob = cp.Problem(cp.Minimize(obj), const)
    prob.solve(solver=cp.ECOS)
    return lambda_dual_matrix.value