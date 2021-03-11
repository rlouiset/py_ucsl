from sklearn.base import RegressorMixin

from sklearn.metrics import adjusted_rand_score as ARI
from EM_HYDRA.utils import *
from EM_HYDRA.base import *

import logging
import copy


class UCSL_R(BaseEM, RegressorMixin):
    """UCSL regressor.
    Implementation of Mike Tipping"s Relevance Vector Machine for
    classification using the scikit-learn API.

    Parameters
    ----------
    C : float, optional (default=1)
        SVM tolerance parameter (Maximization step), if too tiny, risk of overfit.
        If none is given, 1 will be used.
    initialization : string, optional (default="DPP")
        Initialization of each consensus run,
        If not specified, "Determinental Point Process" will be used.
    clustering : string, optional (default="original")
        Clustering method for the Expectation step,
        It must be one of "original", "k_means", "gaussian_mixture".
        If not specified, HYDRA original "Max Margin Distance" will be used.
    consensus : string, optional (default="spectral_clustering")
        Consensus method for the Clustering bagging method,
        If not specified, HYDRA original "Spectral Clustering" will be used.
    weighting : string, optional (default="spectral_clustering")
        weighting method during the whole algorithm processing,
        It must be one of "soft_clustering", "hard_clustering".
        ie : the importance of non-clustered label in the maximization computation
        If not specified, HYDRA original "all" will be used.
    """

    def __init__(self, stability_threshold=0.95, noise_tolerance_threshold=10, C=1, n_consensus=10, n_iterations=10,
                 initialization="gaussian_mixture", clustering='gaussian_mixture', consensus='spectral_clustering',
                 maximization='svr', custom_clustering_method=None, custom_maximization_method=None, n_clusters=2,
                 weighting='soft_clustering', custom_initialization_matrixes=None):

        super().__init__(initialization=initialization, clustering=clustering, consensus=consensus, maximization=maximization,
                         stability_threshold=stability_threshold, noise_tolerance_threshold=noise_tolerance_threshold,
                         custom_clustering_method=custom_clustering_method,
                         custom_maximization_method=custom_maximization_method,
                         custom_initialization_matrixes=custom_initialization_matrixes,
                         n_consensus=n_consensus, n_iterations=n_iterations)

        # define C hyper-parameter if the classification method is max-margin
        self.C = C

        # define what are the weightings we want=
        assert (weighting in ['hard_clustering', 'soft_clustering']), \
            "weighting must be one of 'hard_clustering', 'soft_clustering'"
        self.weighting = weighting

        # define the number of clusters
        if n_clusters is not None :
            self.n_clusters = n_clusters
            self.adaptive_clustering = False
        else :
            self.n_clusters = 8
            self.adaptive_clustering = False

        # store directions from the Maximization method and store intercepts (only useful for HYDRA)
        self.coefficients = {cluster_i: [] for cluster_i in range(self.n_clusters)}
        self.intercepts = {cluster_i: [] for cluster_i in range(self.n_clusters)}

        # store intermediate and consensus results in dictionaries
        self.cluster_labels_ = None
        self.clustering_assignments = None
        # define barycenters saving dictionaries
        self.barycenters = None

        # define orthonormal directions basis and clustering methods at each consensus step
        self.orthonormal_basis = {c: {} for c in range(n_consensus)}
        self.clustering_method = {c: {} for c in range(n_consensus)}

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
        # cluster each label one by one and confine the other inside the polytope
        self.run(X_train, y_train, self.n_clusters)
        return self

    def predict(self, X):
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
        y_pred = np.zeros((len(X),))

        if self.maximization in ['svr']:
            predictions_per_clusters = self.compute_predictions_per_clusters(X)
            # compute the predictions \w.r.t cluster previously found
            cluster_predictions = self.predict_clusters(X)
            y_pred = sum([cluster_predictions[:, cluster] * predictions_per_clusters[:, cluster] for cluster in range(self.n_clusters)])

        return y_pred

    def compute_predictions_per_clusters(self, X):
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
        SVM_predictions = np.zeros((len(X), self.n_clusters))

        for cluster_i in range(self.n_clusters):
            SVM_coefficient = self.coefficients[cluster_i]
            SVM_intercept = self.intercepts[cluster_i]
            SVM_predictions[:, cluster_i] = X @ SVM_coefficient[0] + SVM_intercept[0]
        return SVM_predictions

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
        cluster_predictions = np.zeros((len(X), self.n_clusters))

        if self.clustering in ['k_means', 'gaussian_mixture', 'custom']:
            if self.n_clusters > 1:
                X_proj = X @ self.orthonormal_basis[-1].T
                if self.clustering == 'k_means':
                    cluster_predictions = one_hot_encode(self.clustering_method[-1].predict(X_proj).astype(np.int), n_classes=self.n_clusters)
                elif self.clustering == 'gaussian_mixture':
                    cluster_predictions = self.clustering_method[-1].predict_proba(X_proj)
                elif self.clustering == 'custom':
                    Q_distances = np.zeros((len(X_proj), len(self.barycenters)))
                    for cluster in range(len(self.barycenters)):
                        if X_proj.shape[1] > 1:
                            Q_distances[:, cluster] = np.sum(np.abs(X_proj - self.barycenters[cluster]), 1)
                        else:
                            Q_distances[:, cluster] = (X_proj - self.barycenters[cluster][None, :])[:, 0]
                    Q_distances /= np.sum(Q_distances, 1)[:, None]
                    cluster_predictions = 1 - Q_distances
            else:
                cluster_predictions = np.ones((len(X), 1))
        return cluster_predictions

    def run(self, X, y, n_clusters):
        if n_clusters == 1:
            # by default, when we do not want to cluster a label, we train a simple linear SVM
            SVM_coefficient, SVM_intercept = launch_svr(X, y, C=self.C)
            self.coefficients[0] = SVM_coefficient
            self.intercepts[0] = SVM_intercept
            n_consensus = 0
        else:
            n_consensus = self.n_consensus
            # define the clustering assignment matrix (each column correspond to one consensus run)
            self.clustering_assignments = np.zeros((len(y), n_consensus))

        for consensus in range(n_consensus):
            # first we initialize the clustering matrix S, with the initialization strategy set in self.initialization
            S, cluster_index, n_clusters = self.initialize_clustering(X, y, n_clusters)
            if self.weighting in ['hard_clustering']:
                S = np.rint(S)

            cluster_index = self.run_EM(X, y, S, cluster_index, n_clusters, self.stability_threshold, consensus)

            # update the cluster index for the consensus clustering
            self.clustering_assignments[:, consensus] = cluster_index

        if n_consensus > 1:
            self.clustering_bagging(X, y, n_clusters)

    def initialize_clustering(self, X, y_polytope, n_clusters):
        """Perform a bagging of the previously obtained clusterings and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        n_clusters : int
            number of clusters to be set.
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        S = np.ones((len(y_polytope), n_clusters)) / n_clusters

        if self.initialization in ["k_means"]:
            KM = KMeans(n_clusters=self.n_clusters, n_init=1).fit(X)
            S = one_hot_encode(KM.predict(X))

        if self.initialization in ["gaussian_mixture"]:
            GMM = GaussianMixture(n_components=self.n_clusters, n_init=1).fit(X)
            S = GMM.predict_proba(X)

        if self.initialization in ['custom']:
            custom_clustering_method_ = copy.deepcopy(self.custom_clustering_method)
            S = one_hot_encode(custom_clustering_method_.fit_predict(X), n_classes=n_clusters)

        if self.initialization == "precomputed":
            S = self.custom_initialization_matrixes

        cluster_index = np.argmax(S, axis=1)

        if self.adaptive_clustering :
            n_clusters = max(S.shape[1], 2)

        return S, cluster_index, n_clusters

    def maximization_step(self, X, y, S, n_clusters, iteration):
        if self.maximization == "logistic":
            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                logistic_coefficient, logistic_intercept = launch_logistic(X, y, cluster_assignment)
                self.coefficients[cluster].extend(logistic_coefficient)
                self.intercepts[cluster] = logistic_intercept
        if self.maximization == "svr":
            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                logistic_coefficient, logistic_intercept = launch_svr(X, y, cluster_assignment)
                self.coefficients[cluster].extend(logistic_coefficient)
                self.intercepts[cluster] = logistic_intercept

    def expectation_step(self, X, S, n_clusters, consensus):
        """Update clustering method (update clustering distribution matrix S).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.

        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        n_clusters : int
            the number of clusters
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
        if self.clustering == 'HYDRA':
            SVM_distances = np.zeros(S.shape)
            for cluster in range(n_clusters):
                # Apply the data again the trained model to get the final SVM scores
                SVM_coefficient = self.coefficients[cluster]
                SVM_intercept = self.intercepts[cluster]
                SVM_distances[:, cluster] = X @ SVM_coefficient[0] + SVM_intercept[0]

            SVM_distances -= np.min(SVM_distances)
            SVM_distances += 1e-3
            Q = SVM_distances / np.sum(SVM_distances, 1)[:, None]

        if self.clustering in ['k_means', 'gaussian_mixture', 'custom']:
            # get directions
            directions = []
            for cluster in range(n_clusters):
                directions.extend(self.coefficients[cluster])
            norm_directions = [np.linalg.norm(direction) for direction in directions]
            directions = np.array(directions) / np.array(norm_directions)[:, None]

            # compute the most important vectors because Graam-Schmidt is not invariant by permutation when the matrix is not square
            scores = []
            for i, direction_i in enumerate(directions):
                scores_i = []
                for j, direction_j in enumerate(directions):
                    if i != j:
                        scores_i.append(np.linalg.norm(direction_i - (np.dot(direction_i, direction_j) * direction_j)))
                scores.append(np.mean(scores_i))
            directions = directions[np.array(scores).argsort()[::-1], :]

            # orthonormalize coefficient/direction basis
            basis = []
            for v in directions:
                w = v - np.sum(np.dot(v, b) * b for b in basis)
                if len(basis) >= 2:
                    if np.linalg.norm(w) * self.noise_tolerance_threshold > 1:
                        basis.append(w / np.linalg.norm(w))
                elif np.linalg.norm(w) > 1e-2:
                    basis.append(w / np.linalg.norm(w))

            self.orthonormal_basis[consensus] = np.array(basis)
            self.orthonormal_basis[-1] = np.array(basis).copy()
            X_proj = X @ self.orthonormal_basis[consensus].T

            centroids = [np.mean(S[:, cluster][:, None] * X_proj, 0) for cluster in range(n_clusters)]

            if self.clustering == 'k_means':
                self.clustering_method[consensus] = KMeans(
                    n_clusters=n_clusters, init=np.array(centroids), n_init=1).fit(X_proj)
                Q = one_hot_encode(self.clustering_method[consensus].predict(X_proj), n_classes=n_clusters)
                self.clustering_method[-1] = copy.deepcopy(
                    self.clustering_method[consensus])

            if self.clustering == 'gaussian_mixture':
                self.clustering_method[consensus] = GaussianMixture(
                    n_components=n_clusters, covariance_type='full', means_init=np.array(centroids)).fit(X_proj)
                Q = self.clustering_method[consensus].predict_proba(X_proj)
                self.clustering_method[-1] = copy.deepcopy(self.clustering_method[consensus])

            if self.clustering in ['custom']:
                self.clustering_method[consensus] = copy.deepcopy(self.custom_clustering_method)
                Q = one_hot_encode(self.clustering_method[consensus].fit_predict(X_proj), n_clusters)

        # define matrix clustering
        S = Q.copy()
        cluster_index = np.argmax(Q, axis=1)

        if self.adaptive_clustering :
            n_clusters = max(S.shape[1], 2)

        return S, cluster_index, n_clusters

    def run_EM(self, X, y, S, cluster_index, n_clusters, stability_threshold, consensus):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Original target values.
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        cluster_index : array-like, shape (n_positives_samples, )
            clusters predictions argmax for positive samples.
        n_clusters : int
            number of clusters to be set.
        stability_threshold : float
            stability threshold where we stopped the algorithm when we reach it
        consensus : int
            index of consensus
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        for iteration in range(self.n_iterations):
            # check for degenerate clustering for positive labels (warning) and negatives (might be normal)
            for cluster in range(self.n_clusters):
                if np.count_nonzero(S[:, cluster]) == 0:
                    logging.debug("Cluster dropped, one cluster have no positive points anymore, in iteration : %d" % (iteration - 1))
                    logging.debug("Re-initialization of the clustering...")
                    S, cluster_index, n_clusters = self.initialize_clustering(X, y, n_clusters)

            # re-init directions for each clusters
            self.coefficients = {cluster_i: [] for cluster_i in range(n_clusters)}
            self.intercepts = {cluster_i: [] for cluster_i in range(n_clusters)}

            # maximizes likelihood
            self.maximization_step(X, y, S, n_clusters, iteration)

            # decide the convergence based on the clustering stability
            S_hold = S.copy()
            S, cluster_index, n_clusters = self.expectation_step(X, S, n_clusters, consensus)

            # applying the weighting set as input
            if self.weighting in ['hard_clustering']:
                S = np.rint(S)

            # check the Clustering Stability \w Adjusted Rand Index for stopping criteria
            cluster_consistency = ARI(np.argmax(S, 1), np.argmax(S_hold, 1))
            print(cluster_consistency)
            if cluster_consistency > stability_threshold:
                break
        print('')
        return cluster_index

    def predict_clusters_proba_from_cluster_labels(self, X, n_clusters):
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
            X_proj = X @ self.orthonormal_basis[consensus].T
            if self.clustering in ['k_means', 'gaussian_mixture']:
                X_clustering_assignments[:, consensus] = self.clustering_method[consensus].predict(X_proj)
            elif self.clustering in ['custom']:
                X_clustering_assignments[:, consensus] = self.clustering_method[consensus].fit_predict(X_proj)
        similarity_matrix = compute_similarity_matrix(self.clustering_assignments, clustering_assignments_to_pred=X_clustering_assignments)

        Q = np.zeros((len(X), n_clusters))
        y_clusters_train_ = self.cluster_labels_
        for cluster in range(n_clusters):
            Q[:, cluster] = np.mean(similarity_matrix[y_clusters_train_ == cluster], 0)
        Q /= np.sum(Q, 1)[:, None]
        return Q

    def clustering_bagging(self, X, y, n_clusters):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Original target values.
        n_clusters : int
            number of clusters to be set.
        Returns
        -------
        None
        """
        # perform consensus clustering
        consensus_cluster_index = compute_spectral_clustering_consensus(self.clustering_assignments, n_clusters)
        # save clustering predictions computed by bagging step
        self.cluster_labels_ = consensus_cluster_index

        # update clustering matrix S
        S = self.predict_clusters_proba_from_cluster_labels(X, n_clusters)
        if self.weighting in ['hard_clustering']:
            S = np.rint(S)

        cluster_index = self.run_EM(X, y, S, consensus_cluster_index, n_clusters, 0.99, -1)

        # save barycenters and final predictions
        self.cluster_labels_ = cluster_index
        X_proj = X @ self.orthonormal_basis[-1].T
        self.barycenters = [np.mean(X_proj[cluster_index == cluster], 0)[None, :] for cluster in range(np.max(cluster_index) + 1)]
