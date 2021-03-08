from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta, abstractmethod

from sklearn.metrics import adjusted_rand_score as ARI
from EM_HYDRA.DPP_utils import *
from EM_HYDRA.utils import *
from sklearn.svm import SVC

import logging
import copy


class BaseEM(BaseEstimator, metaclass=ABCMeta):
    """Basic class for our Machine Learning Expectation-Maximization framework."""

    @abstractmethod
    def __init__(self, C, kernel, stability_threshold, noise_tolerance_threshold,
                 n_consensus, n_iterations, n_labels, n_clusters_per_label,
                 initialization, clustering, consensus, negative_weighting):

        if stability_threshold < 0 or stability_threshold > 1:
            msg = "The stability_threshold value is invalid. It must be between 0 and 1."
            raise ValueError(msg)

        # define numerical hyperparameters
        self.C = C
        self.kernel = kernel
        self.stability_threshold = stability_threshold
        self.noise_tolerance_threshold = noise_tolerance_threshold

        # define number of iterations or consensus to perform
        self.n_consensus = n_consensus
        self.n_iterations = n_iterations

        # define n_labels and n_clusters per label
        self.n_labels = n_labels
        if n_clusters_per_label is None:
            self.n_clusters_per_label = {label: 2 for label in range(n_labels)}
        else:
            self.n_clusters_per_label = n_clusters_per_label

        # define what type of initialization, clustering and consensus one wants to use
        self.initialization = initialization
        self.clustering = clustering
        self.consensus = consensus
        self.negative_weighting = negative_weighting


class HYDRA(BaseEM, ClassifierMixin):
    """Relevance Vector Classifier.
    Implementation of Mike Tipping"s Relevance Vector Machine for
    classification using the scikit-learn API.

    Parameters
    ----------
    C : float, optional (default=1)
        SVM tolerance parameter (Maximization step), if too tiny, risk of overfit.
        If none is given, 1 will be used.
    kernel : string, optional (default="linear")
        Specifies the kernel type to be used in the algorithm.
        It must be one of "linear", "poly", "rbf", "sigmoid" or ‘precomputed’.
        If none is given, "linear" will be used.
    initialization : string, optional (default="DPP")
        Initialization of each consensus run,
        If not specified, "Determinental Point Process" will be used.
    clustering : string, optional (default="original")
        Clustering method for the Expectation step,
        It must be one of "original", "boundary_barycenter", "k_means", "gaussian_mixture" or "bissector_hyperplane".
        If not specified, HYDRA original "Max Margin Distance" will be used.
    consensus : string, optional (default="spectral_clustering")
        Consensus method for the Clustering bagging method,
        If not specified, HYDRA original "Spectral Clustering" will be used.
    negative_weighting : string, optional (default="spectral_clustering")
        negative_weighting method during the whole algorithm processing,
        It must be one of "all", "soft_clustering", "hard_clustering".
        ie : the importance of non-clustered label in the SVM computation
        If not specified, HYDRA original "all" will be used.
    """

    def __init__(self, C=1, kernel="linear", stability_threshold=0.95, noise_tolerance_threshold=5,
                 n_consensus=5, n_iterations=5, n_labels=2, n_clusters_per_label=None,
                 initialization="DPP", clustering='original', consensus='spectral_clustering', negative_weighting='all',
                 training_label_mapping=None, initialization_matrixes=None):

        super().__init__(C=C, kernel=kernel, stability_threshold=stability_threshold,
                         noise_tolerance_threshold=noise_tolerance_threshold,
                         n_consensus=n_consensus, n_iterations=n_iterations, n_labels=n_labels,
                         n_clusters_per_label=n_clusters_per_label,
                         initialization=initialization, clustering=clustering, consensus=consensus,
                         negative_weighting=negative_weighting)

        # define the mapping of labels before fitting the algorithm
        # for example, one may want to merge 2 labels together before fitting to check if clustering separate them well
        if training_label_mapping is None:
            self.training_label_mapping = {label: label for label in range(self.n_labels)}
        else:
            self.training_label_mapping = training_label_mapping

        # define clustering parameters
        self.barycenters = {label: {cluster: None for cluster in range(n_clusters_per_label[label])} for label in
                            range(self.n_labels)}
        self.coefficients = {label: {cluster_i: None for cluster_i in range(n_clusters_per_label[label])} for label in
                             range(self.n_labels)}
        self.intercepts = {label: {cluster_i: None for cluster_i in range(n_clusters_per_label[label])} for label in
                           range(self.n_labels)}

        # TODO : Get rid of these visualization helps
        self.S_lists = {label: dict() for label in range(self.n_labels)}
        self.coef_lists = {label: {cluster_i: dict() for cluster_i in range(n_clusters_per_label[label])} for label in
                           range(self.n_labels)}
        self.intercept_lists = {label: {cluster_i: dict() for cluster_i in range(n_clusters_per_label[label])} for label
                                in range(self.n_labels)}

        # define bissector hyperplane parameter
        self.mean_direction = {label: None for label in range(self.n_labels)}

        # define k_means clustering method orthonormal basis and k_means
        self.y_clusters_pred = {label: None for label in range(self.n_labels)}
        self.orthonormal_basis = {label: [None for consensus in range(n_consensus)] for label in range(self.n_labels)}
        self.k_means = {label: [None for consensus in range(n_consensus)] for label in range(self.n_labels)}
        self.gaussian_mixture = {label: [None for consensus in range(n_consensus)] for label in range(self.n_labels)}

        self.clustering_assignments = {label: None for label in range(self.n_labels)}

        self.initialization_matrixes = initialization_matrixes

    def fit(self, X_train, y_train):
        """Fit the HYDRA model according to the given training data.
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training vectors.
        y_train : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self
        """
        # apply label mapping (in our case we merged "BIPOLAR" and "SCHIZOPHRENIA" into "MENTAL DISEASE" for our xp)
        y_train_copy = y_train.copy()
        for original_label, new_label in self.training_label_mapping.items():
            y_train_copy[y_train == original_label] = new_label

        # cluster each label one by one and confine the other inside the polytope
        for label in range(self.n_labels):
            self.run(X_train, y_train_copy, self.n_clusters_per_label[label], idx_outside_polytope=label)

        return self

    def predict(self, X):
        """Predict using the HYDRA model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query points to be evaluate.
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predictions of the labels of the query points.
        """
        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, 1)

    def predict_proba(self, X):
        """Predict using the HYDRA model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query points to be evaluate.
        Returns
        -------
        y_pred : array, shape (n_samples, n_labels)
            Predictions of the probabilities of the query points belonging to labels.
        """
        y_pred = np.zeros((len(X), self.n_labels))
        SVM_distances = self.compute_distances_to_hyperplanes(X)

        if self.clustering in ['original']:
            # merge each label distances and compute the probability \w sigmoid function
            if self.n_labels == 2:
                y_pred[:, 1] = sigmoid(np.max(SVM_distances[1], 1) - np.max(SVM_distances[0], 1))
                y_pred[:, 0] = 1 - y_pred[:, 1]
            else:
                for label in range(self.n_labels):
                    y_pred[:, label] = np.max(SVM_distances[label], 1)
                y_pred = py_softmax(y_pred, axis=1)

        else:
            # compute the predictions \w.r.t cluster previously found
            cluster_predictions = self.predict_clusters(X)
            if self.n_labels == 2:
                y_pred[:, 1] = sum(
                    [np.rint(cluster_predictions[1])[:, cluster] * SVM_distances[1][:, cluster] for cluster in
                     range(self.n_clusters_per_label[1])])
                # y_pred[:, 1] -= sum([cluster_predictions[0][:, cluster] * SVM_distances[0][:, cluster] for cluster in
                #                     range(self.n_clusters_per_label[0])])
                # compute probabilities \w sigmoid
                y_pred[:, 1] = sigmoid(y_pred[:, 1] / np.max(y_pred[:, 1]))
                y_pred[:, 0] = 1 - y_pred[:, 1]
            else:
                for label in range(self.n_labels):
                    y_pred[:, label] = sum(
                        [cluster_predictions[label][:, cluster] * SVM_distances[label][:, cluster] for cluster in
                         range(self.n_clusters_per_label[label])])
                y_pred = py_softmax(y_pred, axis=1)

        return y_pred

    def compute_distances_to_hyperplanes(self, X):
        """Predict using the HYDRA model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query points to be evaluate.
        Returns
        -------
        SVM_distances : dict of array, length (n_labels) , shape of element (n_samples, n_clusters[label])
            Predictions of the point/hyperplane margin for each cluster of each label.
        """
        # first compute points distances to hyperplane
        SVM_distances = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in range(self.n_labels)}

        for label in range(self.n_labels):
            for cluster_i in range(self.n_clusters_per_label[label]):
                SVM_coefficient = self.coefficients[label][cluster_i]
                SVM_intercept = self.intercepts[label][cluster_i]
                SVM_distances[label][:, cluster_i] = X @ SVM_coefficient[0] + SVM_intercept[0]
        return SVM_distances

    def predict_clusters(self, X):
        """Predict clustering for each label in a hierarchical manner.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        Returns
        -------
        cluster_predictions : dict of arrays, length (n_labels) , shape per key:(n_samples, n_clusters[key])
            Dict containing clustering predictions for each label, the dictionary keys are the labels
        """
        cluster_predictions = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in
                               range(self.n_labels)}

        if self.clustering in ['original']:
            SVM_distances = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in
                             range(self.n_labels)}

            for label in range(self.n_labels):
                # fullfill the SVM score matrix
                for cluster in range(self.n_clusters_per_label[label]):
                    SVM_coefficient = self.coefficients[label][cluster]
                    SVM_intercept = self.intercepts[label][cluster]
                    SVM_distances[label][:, cluster] = 1 + X @ SVM_coefficient[0] + SVM_intercept[0]

                # compute clustering conditional probabilities as in the original HYDRA paper : P(cluster=i|y=label)
                SVM_distances[label] -= np.min(SVM_distances[label])
                SVM_distances[label] += 1e-3

            for label in range(self.n_labels):
                cluster_predictions[label] = SVM_distances[label] / np.sum(SVM_distances[label], 1)[:, None]

        elif self.clustering in ['boundary_barycenter']:
            barycenters_distances = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in
                                     range(self.n_labels)}
            for label in range(self.n_labels):
                for cluster in range(self.n_clusters_per_label[label]):
                    # get directions, intercepts of SVMs
                    w_cluster = self.coefficients[label][cluster]
                    b_cluster = self.intercepts[label][cluster]
                    w_cluster_normed = w_cluster / np.linalg.norm(w_cluster) ** 2

                    # project barycenter point on the boundary
                    boundary_barycenter = self.barycenters[label][cluster] + (
                            self.barycenters[label][cluster] @ w_cluster[0] + b_cluster) * w_cluster_normed

                    # compute distance to barycenter and assign cluster to closest barycenter : P(cluster=i|y=label)
                    barycenters_distances[label][:, cluster] = -np.linalg.norm(X - boundary_barycenter, axis=1)
                    barycenters_distances[label][:, cluster] = sigmoid(
                        barycenters_distances[label][:, cluster] / np.max(barycenters_distances[label][:, cluster]))

                for cluster in range(self.n_clusters_per_label[label]):
                    cluster_predictions[label][:, cluster] = barycenters_distances[label][:, cluster] / np.sum(
                        barycenters_distances[label], 1)

        elif self.clustering in ['bisector_hyperplane']:
            for label in range(self.n_labels):
                X_proj = X @ self.mean_direction[label][:, None]
                y_proj_pred = sigmoid(X_proj[:, None] / np.max(X_proj))

                cluster_predictions[label][:, 0] = (1 - y_proj_pred)
                cluster_predictions[label][:, 1] = y_proj_pred

        elif self.clustering in ['k_means', 'gaussian_mixture']:
            for label in range(self.n_labels):
                if self.n_clusters_per_label[label] > 1:
                    X_proj = X @ self.orthonormal_basis[label][-1].T
                    if self.clustering == 'k_means':
                        cluster_predictions[label] = one_hot_encode(
                            self.k_means[label][-1].predict(X_proj).astype(np.int),
                            n_classes=self.n_clusters_per_label[label])
                    elif self.clustering == 'gaussian_mixture':
                        cluster_predictions[label] = self.gaussian_mixture[label][-1].predict_proba(X_proj)
                else:
                    cluster_predictions[label] = np.ones((len(X), 1))
        return cluster_predictions

    def run(self, X, y, n_clusters, idx_outside_polytope):
        # set label idx_outside_polytope outside of the polytope by setting it to positive labels
        y_polytope = np.copy(y)
        # if label is inside of the polytope, the distance is negative and the label is not divided into
        y_polytope[y_polytope != idx_outside_polytope] = -1
        # if label is outside of the polytope, the distance is positive and the label is clustered
        y_polytope[y_polytope == idx_outside_polytope] = 1

        index_positives = np.where(y_polytope == 1)[0]  # index for Positive labels (outside polytope)
        index_negatives = np.where(y_polytope == -1)[0]  # index for Negative labels (inside polytope)

        if n_clusters == 1:
            SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, kernel='rbf')
            self.coefficients[idx_outside_polytope][0] = SVM_coefficient
            self.intercepts[idx_outside_polytope][0] = SVM_intercept
            n_consensus = 0
        else:
            n_consensus = self.n_consensus
            # define the clustering assignment matrix (each column correspond to one consensus run)
            self.clustering_assignments[idx_outside_polytope] = np.zeros((len(index_positives), n_consensus))

        for consensus in range(n_consensus):
            # first we initialize the clustering matrix S, with the initialization strategy set in self.initialization
            S, cluster_index = self.initialize_clustering(X, y_polytope, index_positives, index_negatives,
                                                          n_clusters, idx_outside_polytope)
            # TODO : Get rid of these visualization helps
            self.S_lists[idx_outside_polytope][0] = S.copy()

            self.run_EM(X, y_polytope, S, cluster_index, index_positives, index_negatives, idx_outside_polytope,
                        n_clusters, consensus)

            # update the cluster index for the consensus clustering
            self.clustering_assignments[idx_outside_polytope][:, consensus] = cluster_index + 1

        if n_consensus > 0:
            self.clustering_bagging(X, y_polytope, index_positives, index_negatives, idx_outside_polytope, n_clusters)

    def predict_clusters_proba_for_negative_points(self, X, idx_outside_polytope, n_clusters):
        """Predict positive and negative points clustering probabilities.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        idx_outside_polytope : int
            label that is being clustered
        n_clusters :
            number of clusters
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        X_clustering_assignments = np.zeros((len(X), self.n_consensus))
        for consensus in range(self.n_consensus):
            X_proj = X @ self.orthonormal_basis[idx_outside_polytope][consensus].T
            if self.clustering == 'k_means':
                X_clustering_assignments[:, consensus] = self.k_means[idx_outside_polytope][consensus].predict(X_proj)
            elif self.clustering == 'gaussian_mixture':
                X_clustering_assignments[:, consensus] = self.gaussian_mixture[idx_outside_polytope][consensus].predict(
                    X_proj)
        similarity_matrix = compute_similarity_matrix(self.clustering_assignments[idx_outside_polytope],
                                                      clustering_assignments_to_pred=X_clustering_assignments)

        Q = np.zeros((len(X), n_clusters))
        y_clusters_train_ = self.y_clusters_pred[idx_outside_polytope]
        for cluster in range(n_clusters):
            Q[:, cluster] = np.mean(similarity_matrix[y_clusters_train_ == cluster], 0)
        Q /= np.sum(Q, 1)[:, None]
        return Q

    def initialize_clustering(self, X, y_polytope, index_positives, index_negatives, n_clusters, idx_outside_polytope):
        """Perform a bagging of the previously obtained clusterings and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        index_negatives : array-like, shape (n_negatives_samples, )
            indexes of the negatives labels nàt being clustered
        n_clusters : int
            number of clusters to be set.
        idx_outside_polytope : int
            label that is being clustered
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        S = np.ones((len(y_polytope), n_clusters)) / n_clusters

        if self.initialization == "DPP":
            num_subject = y_polytope.shape[0]
            W = np.zeros((num_subject, X.shape[1]))
            for j in range(num_subject):
                ipt = np.random.randint(len(index_positives))
                icn = np.random.randint(len(index_negatives))
                W[j, :] = X[index_positives[ipt], :] - X[index_negatives[icn], :]

            KW = np.matmul(W, W.transpose())
            KW = np.divide(KW, np.sqrt(np.multiply(np.diag(KW)[:, np.newaxis], np.diag(KW)[:, np.newaxis].transpose())))
            evalue, evector = np.linalg.eig(KW)
            Widx = sample_dpp(np.real(evalue), np.real(evector), n_clusters)
            prob = np.zeros((len(index_positives), n_clusters))  # only consider the PTs

            for i in range(n_clusters):
                prob[:, i] = np.matmul(
                    np.multiply(X[index_positives, :],
                                np.divide(1, np.linalg.norm(X[index_positives, :], axis=1))[:, np.newaxis]),
                    W[Widx[i], :].transpose())

            prob = py_softmax(prob, 1)
            S[index_positives] = cpu_sk(prob, 1)

        if self.initialization == "k_means":
            KM = KMeans(n_clusters=self.n_clusters_per_label[idx_outside_polytope]).fit(X[index_positives])
            S = one_hot_encode(KM.predict(X))

        if self.initialization == "gaussian_mixture":
            GMM = GaussianMixture(n_components=self.n_clusters_per_label[idx_outside_polytope]).fit(X[index_positives])
            S = GMM.predict_proba(X)

        if self.initialization == "precomputed":
            S = self.initialization_matrixes[idx_outside_polytope]

        cluster_index = np.argmax(S[index_positives], axis=1)
        return S, cluster_index

    def update_clustering(self, X, S, index_positives, cluster_index, n_clusters, idx_outside_polytope, consensus):
        """Update clustering method (update clustering distribution matrix S).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.

        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        cluster_index : array-like, shape (n_positives_samples, )
            clusters predictions argmax for positive samples.
        n_clusters : int
            number of clusters to be set.
        idx_outside_polytope : int
            label that is being clustered
        consensus : int
            which consensus is being run ?
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        cluster_index : array-like, shape (n_positives_samples, )
            clusters predictions argmax for positive samples.
        """
        Q = S.copy()
        if self.clustering == 'original':
            SVM_distances = np.zeros(S.shape)
            for cluster in range(self.n_clusters_per_label[idx_outside_polytope]):
                # Apply the data again the trained model to get the final SVM scores
                SVM_coefficient = self.coefficients[idx_outside_polytope][cluster]
                SVM_intercept = self.intercepts[idx_outside_polytope][cluster]
                SVM_distances[:, cluster] = X @ SVM_coefficient[0] + SVM_intercept[0]

            SVM_distances -= np.min(SVM_distances)
            SVM_distances += 1e-3
            Q = SVM_distances / np.sum(SVM_distances, 1)[:, None]

        if self.clustering in ['k_means', 'gaussian_mixture']:
            directions = [self.coefficients[idx_outside_polytope][cluster_i][0] for cluster_i in
                          range(self.n_clusters_per_label[idx_outside_polytope])]
            norm_directions = [np.linalg.norm(direction) for direction in directions]
            directions = np.array(directions) / np.array(norm_directions)[:, None]

            # compute the most important vectors because Graam Schmidt is not invariant by permutation when the matrix is not square
            scores = []
            for i, direction_i in enumerate(directions):
                scores_i = []
                for j, direction_j in enumerate(directions):
                    if i != j:
                        scores_i.append(np.linalg.norm(direction_i - (np.dot(direction_i, direction_j) * direction_j)))
                scores.append(np.mean(scores_i))
            directions = directions[np.array(scores).argsort()[::-1], :]

            basis = []
            for v in directions:
                w = v - np.sum(np.dot(v, b) * b for b in basis)
                if len(basis) >= 2 :
                    if np.linalg.norm(w) * self.noise_tolerance_threshold > 1 :
                        basis.append(w / np.linalg.norm(w))
                elif np.linalg.norm(w) > 1e-2:
                    basis.append(w / np.linalg.norm(w))

            self.orthonormal_basis[idx_outside_polytope][consensus] = np.array(basis)
            self.orthonormal_basis[idx_outside_polytope][-1] = np.array(basis).copy()
            X_proj = X @ self.orthonormal_basis[idx_outside_polytope][consensus].T

            centroids = [np.mean(S[index_positives, cluster_i][:, None] * X_proj[index_positives, :], 0) for cluster_i
                         in
                         range(self.n_clusters_per_label[idx_outside_polytope])]

            if self.clustering == 'k_means':
                self.k_means[idx_outside_polytope][consensus] = KMeans(
                    n_clusters=self.n_clusters_per_label[idx_outside_polytope],
                    init=np.array(centroids), n_init=1).fit(X_proj[index_positives])
                Q = one_hot_encode(self.k_means[idx_outside_polytope][consensus].predict(X_proj),
                                   n_classes=self.n_clusters_per_label[idx_outside_polytope])
                self.k_means[idx_outside_polytope][-1] = copy.deepcopy(self.k_means[idx_outside_polytope][consensus])

            if self.clustering == 'gaussian_mixture':
                self.gaussian_mixture[idx_outside_polytope][consensus] = GaussianMixture(
                    n_components=self.n_clusters_per_label[idx_outside_polytope],
                    covariance_type='spherical', means_init=np.array(centroids)).fit(
                    X_proj[index_positives])
                Q = self.gaussian_mixture[idx_outside_polytope][consensus].predict_proba(X_proj)
                self.gaussian_mixture[idx_outside_polytope][-1] = copy.deepcopy(
                    self.gaussian_mixture[idx_outside_polytope][consensus])

        elif self.clustering in ['bisector_hyperplane']:
            directions = np.array([self.coefficients[idx_outside_polytope][cluster_i][0] for cluster_i in
                                   range(self.n_clusters_per_label[idx_outside_polytope])])

            directions[0] = directions[0] * np.linalg.norm(directions[1]) ** 2 / np.mean(
                (np.linalg.norm(directions, axis=1) ** 2))
            directions[1] = directions[1] * np.linalg.norm(directions[0]) ** 2 / np.mean(
                (np.linalg.norm(directions, axis=1) ** 2))

            mean_direction = (directions[0] - directions[1]) / 2
            mean_intercept = 0

            X_proj = (np.matmul(mean_direction[None, :], X.transpose()) + mean_intercept).transpose().squeeze()
            X_proj = sigmoid(X_proj[:, None] / np.max(X_proj))

            Q = np.concatenate((1 - X_proj, X_proj), axis=1)


        elif self.clustering == 'boundary_barycenter':
            cluster_barycenters = self.barycenters[idx_outside_polytope]
            boundary_barycenters_distances = np.zeros(S.shape)
            for cluster in range(self.n_clusters_per_label[idx_outside_polytope]):
                # get coefficients w and intercept w
                w_cluster = self.coefficients[idx_outside_polytope][cluster]
                b_cluster = self.intercepts[idx_outside_polytope][cluster]
                w_cluster_norm = w_cluster / np.linalg.norm(w_cluster) ** 2
                # project barycenters on hyperplanes
                boundary_barycenter = cluster_barycenters[cluster] - (
                        cluster_barycenters[cluster] @ w_cluster[0] + b_cluster) * w_cluster_norm
                boundary_barycenters_distances[:, cluster] = np.linalg.norm((X - boundary_barycenter), axis=1)
            # assign points to closest barycenter
            Q = - boundary_barycenters_distances + np.max(boundary_barycenters_distances) + 1e-3
            Q = Q / np.sum(Q, 1)[:, None]

        S = Q.copy()
        cluster_index = np.argmax(Q[index_positives], axis=1)
        return S, cluster_index

    def launch_svc(self, X, y, sample_weight=None, kernel=None):
        """Fit the classification SVMs according to the given training data.
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
        SVM_coefficient : array-like, shape (1, n_features)
            The coefficient of the resulting SVM.
        SVM_intercept : array-like, shape (1,)
            The intercept of the resulting SVM.
        """
        if kernel is None:
            kernel = self.kernel

        # fit the different SVM/hyperplanes
        SVM_classifier = SVC(kernel=kernel, C=self.C)
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

    def run_EM(self, X, y_polytope, S, cluster_index, index_positives, index_negatives, idx_outside_polytope,
               n_clusters, consensus):
        """Perform a bagging of the previously obtained clusterings and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        cluster_index : array-like, shape (n_positives_samples, )
            clusters predictions argmax for positive samples.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        index_negatives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        idx_outside_polytope : int
            label that is being clustered
        n_clusters : int
            number of clusters to be set.
        consensus : int
            index of consensus
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        for iteration in range(self.n_iterations):
            # check for degenerate clustering for positive labels (warning) and negatives (might be normal)
            for cluster in range(self.n_clusters_per_label[idx_outside_polytope]):
                if np.count_nonzero(S[index_positives, cluster]) == 0:
                    logging.debug(
                        "Cluster dropped, one cluster have no positive points anymore, in iteration : %d" % (
                                iteration - 1))
                    logging.debug("Re-initialization of the clustering...")
                    S, cluster_index = self.initialize_clustering(X, y_polytope, index_positives, index_negatives,
                                                                  n_clusters, idx_outside_polytope)

                if np.max(S[index_negatives, cluster]) < 0.01:
                    logging.debug(
                        "Cluster too far, one cluster have no negative points anymore, in consensus : %d" % (
                                iteration - 1))
                    logging.debug("Re-distribution of this cluster negative weight to 'all'...")
                    S[index_negatives, cluster] = 1 / n_clusters

            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_assignment)
                self.coefficients[idx_outside_polytope][cluster] = SVM_coefficient
                self.intercepts[idx_outside_polytope][cluster] = SVM_intercept

                # TODO: get rid of
                self.coef_lists[idx_outside_polytope][cluster][iteration + 1] = SVM_coefficient.copy()
                self.intercept_lists[idx_outside_polytope][cluster][iteration + 1] = SVM_intercept.copy()

            # decide the convergence based on the clustering stability
            S_hold = S.copy()
            S, cluster_index = self.update_clustering(X, S, index_positives, cluster_index, n_clusters,
                                                      idx_outside_polytope, consensus)

            # applying the negative weighting set as input
            if self.negative_weighting == 'all':
                S[index_negatives] = 1 / n_clusters
            elif self.negative_weighting == 'hard_clustering':
                S[index_negatives] = np.rint(S[index_negatives]).astype(np.float)

            # always set positive clustering as hard
            S[index_positives] = 0
            S[index_positives, cluster_index] = 1

            # TODO : get rid of
            self.S_lists[idx_outside_polytope][iteration + 1] = S.copy()

            # check the Clustering Stability \w Adjusted Rand Index for stopping criteria
            cluster_consistency = ARI(np.argmax(S[index_positives], 1), np.argmax(S_hold[index_positives], 1))
            if cluster_consistency > self.stability_threshold:
                break

    def clustering_bagging(self, X, y_polytope, index_positives, index_negatives, idx_outside_polytope, n_clusters):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        index_negatives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        idx_outside_polytope : int
            label that is being clustered
        n_clusters : int
            number of clusters to be set.
        Returns
        -------
        None
        """
        # initialize the consensus clustering vector
        S = np.ones((len(X), n_clusters)) / n_clusters

        # perform consensus clustering
        consensus_cluster_index = compute_spectral_clustering_consensus(
            self.clustering_assignments[idx_outside_polytope], n_clusters)
        self.y_clusters_pred[idx_outside_polytope] = consensus_cluster_index

        # update clustering matrix S
        if self.negative_weighting in ['soft_clustering']:
            S = self.predict_clusters_proba_for_negative_points(X, idx_outside_polytope, n_clusters)
        elif self.negative_weighting in ['hard_clustering']:
            S = np.rint(self.predict_clusters_proba_for_negative_points(X, idx_outside_polytope, n_clusters))
        S[index_positives] *= 0
        S[index_positives, consensus_cluster_index] = 1

        if self.clustering == "original":
            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_assignment)
                self.coefficients[idx_outside_polytope][cluster] = SVM_coefficient
                self.intercepts[idx_outside_polytope][cluster] = SVM_intercept

                # TODO: get rid of
                self.coef_lists[idx_outside_polytope][cluster][-1] = SVM_coefficient.copy()
                self.intercept_lists[idx_outside_polytope][cluster][-1] = SVM_intercept.copy()
        else:
            self.run_EM(X, y_polytope, S, consensus_cluster_index, index_positives, index_negatives,
                        idx_outside_polytope, n_clusters, -1)
