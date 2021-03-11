import numpy as np
from scipy.special import logsumexp
import scipy
from EM_HYDRA.sinkornknopp import *
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression


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

    # iterate
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


def consensus_clustering(clustering_results, n_clusters, index_positives):
    S = np.ones((clustering_results.shape[0], n_clusters)) / n_clusters
    co_occurrence_matrix = np.zeros((clustering_results.shape[0], clustering_results.shape[0]))

    for i in range(clustering_results.shape[0] - 1):
        for j in range(i + 1, clustering_results.shape[0]):
            co_occurrence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])

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
    spectral_features = e_vector.real[:, :n_clusters]

    # apply clustering method
    k_means = KMeans(n_clusters=n_clusters, n_init=10, init="k-means++").fit(spectral_features[index_positives])
    S[index_positives] = one_hot_encode(k_means.labels_.astype(np.int), n_classes=n_clusters)

    return S


def compute_similarity_matrix(consensus_assignment, clustering_assignments_to_pred=None):
    # compute inter-samples positive/negative co-occurence matrix
    similarity_matrix = np.zeros((len(consensus_assignment), len(clustering_assignments_to_pred)))
    for i, p_assignment in enumerate(consensus_assignment):
        for j, new_point_assignment in enumerate(clustering_assignments_to_pred):
            similarity_matrix[i, j] = np.sum(p_assignment == new_point_assignment)
    similarity_matrix += 1e-3
    similarity_matrix /= np.max(similarity_matrix)
    return similarity_matrix


def compute_spectral_clustering_consensus(clustering_results, n_clusters):
    # compute positive samples co-occurence matrix
    n_positives = len(clustering_results)
    similarity_matrix = np.zeros((n_positives, n_positives))
    for i in range(n_positives - 1):
        for j in range(i + 1, n_positives):
            similarity_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])
    similarity_matrix = np.add(similarity_matrix, similarity_matrix.transpose())
    similarity_matrix += 1e-3
    similarity_matrix /= np.max(similarity_matrix)

    # initialize spectral clustering method
    spectral_clustering_method = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    spectral_clustering_method.fit(similarity_matrix)

    return spectral_clustering_method.labels_


def launch_svc(X, y, sample_weight=None, kernel='linear', C=1):
    """Fit the classification SVMs according to the given training data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Training sample weights.
    kernel : string,
        kernel used for SVM.
    C : float,
        SVM hyperparameter C
    Returns
    -------
    SVM_coefficient : array-like, shape (1, n_features)
        The coefficient of the resulting SVM.
    SVM_intercept : array-like, shape (1,)
        The intercept of the resulting SVM.
    """

    # fit the different SVM/hyperplanes
    SVM_classifier = SVC(kernel=kernel, C=C)
    SVM_classifier.fit(X, y, sample_weight=sample_weight)

    # get SVM intercept value
    SVM_intercept = SVM_classifier.intercept_

    # get SVM hyperplane coefficient
    if kernel == 'rbf':
        X_support = X[SVM_classifier.support_]
        y_support = y[SVM_classifier.support_]
        SVM_coefficient = SVM_classifier.dual_coef_ @ np.einsum('i,ij->ij', y_support, X_support)
    else:
        SVM_coefficient = SVM_classifier.coef_

    return SVM_coefficient, SVM_intercept


def launch_logistic(X, y, sample_weight=None):
    """Fit the logistic regressions according to the given training data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Training sample weights.
    Returns
    -------
    logistic_coefficient : array-like, shape (1, n_features)
        The coefficient of the resulting logistic regression.
    """

    # fit the different logistic classifier
    logistic = LogisticRegression()
    logistic.fit(X, y, sample_weight=sample_weight)

    # get logistic coefficient and intercept
    logistic_coefficient = logistic.coef_
    logistic_intercept = logistic.intercept_

    return logistic_coefficient, logistic_intercept
