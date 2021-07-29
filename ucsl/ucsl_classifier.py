import copy
import logging

from sklearn.base import ClassifierMixin
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture

from ucsl.base import *
from ucsl.dpp_utils import *
from ucsl.utils import *


class UCSL_C(BaseEM, ClassifierMixin):
    """ucsl classifier.
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
        If not specified, ucsl original "Max Margin Distance" will be used.
    consensus : string, optional (default="spectral_clustering")
        Consensus method for the Clustering bagging method,
        If not specified, ucsl original "Spectral Clustering" will be used.
    negative_weighting : string, optional (default="spectral_clustering")
        negative_weighting method during the whole algorithm processing,
        It must be one of "all", "soft_clustering", "hard_clustering".
        ie : the importance of non-clustered label in the SVM computation
        If not specified, ucsl original "all" will be used.
    """

    def __init__(self, stability_threshold=0.85, noise_tolerance_threshold=10,
                 n_consensus=10, n_iterations=10,
                 n_clusters=2, label_to_cluster=1,
                 clustering='gaussian_mixture', maximization='logistic',
                 negative_weighting='soft_clustering', positive_weighting='hard_clustering',
                 training_label_mapping=None):

        super().__init__(clustering=clustering, maximization=maximization,
                         stability_threshold=stability_threshold, noise_tolerance_threshold=noise_tolerance_threshold,
                         n_consensus=n_consensus, n_iterations=n_iterations)

        # define the number of clusters needed
        self.n_clusters = n_clusters

        # define which label we want to cluster
        self.label_to_cluster = label_to_cluster

        # define the mapping of labels before fitting the algorithm
        # for example, one may want to merge 2 labels together before fitting to check if clustering separate them well
        if training_label_mapping is None:
            self.training_label_mapping = {label: label for label in range(2)}
        else:
            self.training_label_mapping = training_label_mapping

        # define what are the weightings we want for each label
        assert (negative_weighting in ['hard_clustering', 'soft_clustering', 'all']), \
            "negative_weighting must be one of 'hard_clustering', 'soft_clustering'"
        assert (positive_weighting in ['hard_clustering', 'soft_clustering']), \
            "positive_weighting must be one of 'hard_clustering', 'soft_clustering'"
        self.negative_weighting = negative_weighting
        self.positive_weighting = positive_weighting

        # store directions from the Maximization method and store intercepts (only useful for ucsl)
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
        """Fit the ucsl model according to the given training data.
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
        print(self.training_label_mapping)
        # apply label mapping (in our case we merged "BIPOLAR" and "SCHIZOPHRENIA" into "MENTAL DISEASE" for our xp)
        y_train_copy = y_train.copy()
        for original_label, new_label in self.training_label_mapping.items():
            y_train_copy[y_train == original_label] = new_label

        # run the algorithm
        print(self.label_to_cluster)
        self.run(X_train, y_train_copy, idx_outside_polytope=self.label_to_cluster)

        return self

    def predict(self, X):
        """Predict using the ucsl model.
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

    def predict_classif_proba(self, X):
        """Predict using the ucsl model.
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

        if self.maximization in ['max_margin', 'logistic']:
            hp_distances = self.compute_distances_to_hyperplanes(X)
            if self.clustering in ['HYDRA']:
                # merge each label distances and compute the probability \w sigmoid function
                if self.n_labels == 2:
                    y_pred[:, 1] = sigmoid(np.max(hp_distances[1], 1) - np.max(hp_distances[0], 1))
                    y_pred[:, 0] = 1 - y_pred[:, 1]
                else:
                    for label in range(self.n_labels):
                        y_pred[:, label] = np.max(hp_distances[label], 1)
                    y_pred = py_softmax(y_pred, axis=1)

            else:
                # compute the predictions \w.r.t cluster previously found
                cluster_predictions = self.predict_clusters(X)
                if self.n_labels == 2:
                    y_pred[:, 1] = sum(
                        [np.rint(cluster_predictions[1])[:, cluster] * hp_distances[1][:, cluster] for cluster in
                         range(self.n_clusters_per_label[1])])
                    y_pred[:, 1] -= sum([cluster_predictions[0][:, cluster] * hp_distances[0][:, cluster] for cluster in
                                         range(self.n_clusters_per_label[0])])
                    # compute probabilities \w sigmoid
                    y_pred[:, 1] = sigmoid(y_pred[:, 1] / np.max(y_pred[:, 1]))
                    y_pred[:, 0] = 1 - y_pred[:, 1]
                else:
                    for label in range(self.n_labels):
                        y_pred[:, label] = sum(
                            [cluster_predictions[label][:, cluster] * hp_distances[label][:, cluster] for cluster in
                             range(self.n_clusters_per_label[label])])
                    y_pred = py_softmax(y_pred, axis=1)

        return y_pred

    def compute_distances_to_hyperplanes(self, X):
        """Predict using the ucsl model.
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
        SVM_distances = np.zeros((len(X), self.n_clusters))

        for cluster_i in range(self.n_clusters):
            SVM_coefficient = self.coefficients[cluster_i]
            SVM_intercept = self.intercepts[cluster_i]
            SVM_distances[:, cluster_i] = X @ SVM_coefficient[0] + SVM_intercept[0]

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
        cluster_predictions = np.zeros((len(X), self.n_clusters))

        X_proj = X @ self.orthonormal_basis[-1].T

        Q_distances = np.zeros((len(X_proj), len(self.barycenters)))
        for cluster in range(len(self.barycenters)):
            if X_proj.shape[1] > 1:
                Q_distances[:, cluster] = np.sum(np.abs(X_proj - self.barycenters[cluster]), 1)
            else:
                Q_distances[:, cluster] = (X_proj - self.barycenters[cluster][None, :])[:, 0]
        Q_distances /= np.sum(Q_distances, 1)[:, None]
        cluster_predictions = 1 - Q_distances

        return cluster_predictions

    def run(self, X, y, idx_outside_polytope):
        print(np.unique(y))
        # set label idx_outside_polytope outside of the polytope by setting it to positive labels
        y_polytope = np.copy(y)
        # if label is inside of the polytope, the distance is negative and the label is not divided into
        y_polytope[y_polytope != idx_outside_polytope] = -1
        # if label is outside of the polytope, the distance is positive and the label is clustered
        y_polytope[y_polytope == idx_outside_polytope] = 1

        index_positives = np.where(y_polytope == 1)[0]  # index for Positive labels (outside polytope)
        index_negatives = np.where(y_polytope == -1)[0]  # index for Negative labels (inside polytope)

        print(index_positives)

        n_consensus = self.n_consensus
        # define the clustering assignment matrix (each column correspond to one consensus run)
        self.clustering_assignments = np.zeros((len(index_positives), n_consensus))

        for consensus in range(n_consensus):
            # first we initialize the clustering matrix S, with the initialization strategy set in self.initialization
            S, cluster_index, n_clusters = self.initialize_clustering(X, y_polytope, index_positives)
            if self.negative_weighting in ['uniform']:
                S[index_negatives] = 1 / n_clusters
            elif self.negative_weighting in ['hard']:
                S[index_negatives] = np.rint(S[index_negatives])
            if self.positive_weighting in ['hard']:
                S[index_positives] = np.rint(S[index_positives])

            cluster_index = self.run_EM(X, y_polytope, S, cluster_index, index_positives, index_negatives, consensus)

            # update the cluster index for the consensus clustering
            self.clustering_assignments[:, consensus] = cluster_index

        if n_consensus > 1:
            self.clustering_ensembling(X, y_polytope, index_positives, index_negatives)

    def initialize_clustering(self, X, y_polytope, index_positives):
        """Perform a bagging of the previously obtained clusterings and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        S = np.ones((len(y_polytope), self.n_clusters)) / self.n_clusters

        if self.clustering in ["k_means"]:
            KM = KMeans(n_clusters=self.n_clusters, init="random", n_init=1).fit(X[index_positives])
            S = one_hot_encode(KM.predict(X))

        if self.clustering in ["gaussian_mixture"]:
            GMM = GaussianMixture(n_components=self.n_clusters, init_params="random", n_init=1, covariance_type="spherical").fit(X[index_positives])
            S = GMM.predict_proba(X)

        else :
            custom_clustering_method_ = copy.deepcopy(self.clustering)
            S_positives = custom_clustering_method_.fit_predict(X[index_positives])
            S_distances = np.zeros((len(X), np.max(S_positives) + 1))
            for cluster in range(np.max(S_positives) + 1):
                S_distances[:, cluster] = np.sum(np.abs(X - np.mean(X[index_positives][S_positives == cluster], 0)[None, :]), 1)
            S_distances /= np.sum(S_distances, 1)[:, None]
            S = 1 - S

        cluster_index = np.argmax(S[index_positives], axis=1)

        return S, cluster_index

    def maximization_step(self, X, y_polytope, S, n_clusters):
        if self.maximization == "max_margin":
            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                SVM_coefficient, SVM_intercept = launch_svc(X, y_polytope, cluster_assignment)
                self.coefficients[cluster].extend(SVM_coefficient)
                self.intercepts[cluster] = SVM_intercept

        elif self.maximization == "logistic":
            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                logistic_coefficient, logistic_intercept = launch_logistic(X, y_polytope, cluster_assignment)
                self.coefficients[cluster].extend(logistic_coefficient)
                self.intercepts[cluster] = logistic_intercept

    def expectation_step(self, X, S, index_positives, consensus):
        """Update clustering method (update clustering distribution matrix S).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.

        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
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

        # get directions
        directions = []
        for cluster in range(self.n_clusters):
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

        centroids = [np.mean(S[index_positives, cluster][:, None] * X_proj[index_positives, :], 0) for cluster in range(self.n_clusters)]

        if self.clustering == 'k_means':
            self.clustering_method[consensus] = KMeans(
                n_clusters=self.n_clusters, init=np.array(centroids), n_init=1).fit(X_proj[index_positives])
            Q_positives = self.clustering_method[consensus].fit_predict(X_proj[index_positives])
            Q_distances = np.zeros((len(X_proj), np.max(Q_positives) + 1))
            for cluster in range(np.max(Q_positives) + 1):
                Q_distances[:, cluster] = np.sum(np.abs(X_proj - self.clustering_method[consensus].cluster_centers_[cluster]), 1)
            Q_distances = Q_distances / np.sum(Q_distances, 1)[:, None]
            Q = 1 - Q_distances

        if self.clustering == 'gaussian_mixture':
            self.clustering_method[consensus] = GaussianMixture(
                n_components=self.n_clusters, covariance_type="spherical", means_init=np.array(centroids)).fit(X_proj[index_positives])
            Q = self.clustering_method[consensus].predict_proba(X_proj)
            self.clustering_method[-1] = copy.deepcopy(
                self.clustering_method[consensus])

        else :
            self.clustering_method[consensus] = copy.deepcopy(self.clustering)
            Q_positives = self.clustering_method[consensus].fit_predict(X_proj[index_positives])
            Q_distances = np.zeros((len(X_proj), np.max(Q_positives) + 1))
            for cluster in range(np.max(Q_positives) + 1):
                Q_distances[:, cluster] = np.sum(np.abs(X_proj - np.mean(X_proj[index_positives][Q_positives == cluster], 0)[None, :]), 1)
            Q_distances = Q_distances / np.sum(Q_distances, 1)[:, None]
            Q = 1 - Q_distances

        # define matrix clustering
        S = Q.copy()
        cluster_index = np.argmax(Q[index_positives], axis=1)

        return S, cluster_index, self.n_clusters

    def run_EM(self, X, y_polytope, S, cluster_index, index_positives, index_negatives, consensus):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Original target values.
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
        consensus : int
            index of consensus
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        best_cluster_consistency = 1
        if consensus == -1 :
            save_stabler_coefficients = True
            consensus = self.n_consensus + 1
            best_cluster_consistency = 0

        for iteration in range(self.n_iterations):
            # check for degenerate clustering for positive labels (warning) and negatives (might be normal)
            for cluster in range(self.n_clusters):
                if np.count_nonzero(S[index_positives, cluster]) == 0:
                    logging.debug(
                        "Cluster dropped, one cluster have no positive points anymore, in iteration : %d" % (
                                iteration - 1))
                    logging.debug("Re-initialization of the clustering...")
                    S, cluster_index, n_clusters = self.initialize_clustering(X, y_polytope, index_positives)
                if np.max(S[index_negatives, cluster]) < 0.5:
                    logging.debug(
                        "Cluster too far, one cluster have no negative points anymore, in consensus : %d" % (
                                iteration - 1))
                    logging.debug("Re-distribution of this cluster negative weight to 'all'...")
                    S[index_negatives, cluster] = 1 / self.n_clusters

            # re-init directions for each clusters
            self.coefficients = {cluster_i: [] for cluster_i in range(self.n_clusters)}
            self.intercepts = {cluster_i: [] for cluster_i in range(self.n_clusters)}
            # run maximization step
            self.maximization_step(X, y_polytope, S, iteration)

            # decide the convergence based on the clustering stability
            S_hold = S.copy()
            S, cluster_index, n_clusters = self.expectation_step(X, S, index_positives, consensus)

            # applying the negative weighting set as input
            if self.negative_weighting in ['uniform']:
                S[index_negatives] = 1 / n_clusters
            elif self.negative_weighting in ['hard']:
                S[index_negatives] = np.rint(S[index_negatives])
            if self.positive_weighting in ['hard']:
                S[index_positives] = np.rint(S[index_positives])

            # check the Clustering Stability \w Adjusted Rand Index for stopping criteria
            cluster_consistency = ARI(np.argmax(S[index_positives], 1), np.argmax(S_hold[index_positives], 1))

            if cluster_consistency > best_cluster_consistency :
                best_cluster_consistency = cluster_consistency
                self.coefficients[-1] = copy.deepcopy(self.coefficients)
                self.intercepts[-1] = copy.deepcopy(self.intercepts)
                self.orthonormal_basis[-1] = copy.deepcopy(self.orthonormal_basis[consensus])
                self.clustering_method[-1] = copy.deepcopy(self.clustering_method[consensus])
            if cluster_consistency > self.stability_threshold:
                break

        return cluster_index

    def predict_clusters_proba_from_cluster_labels(self, X):
        """Predict positive and negative points clustering probabilities.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
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
            else :
                X_clustering_assignments[:, consensus] = self.clustering_method[consensus].fit_predict(X_proj)
        similarity_matrix = compute_similarity_matrix(self.clustering_assignments, clustering_assignments_to_pred=X_clustering_assignments)

        Q = np.zeros((len(X), self.n_clusters))
        y_clusters_train_ = self.cluster_labels_
        for cluster in range(self.n_clusters):
            Q[:, cluster] = np.mean(similarity_matrix[y_clusters_train_ == cluster], 0)
        Q /= np.sum(Q, 1)[:, None]
        return Q

    def clustering_ensembling(self, X, y_polytope, index_positives, index_negatives):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Modified target values.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        index_negatives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        Returns
        -------
        None
        """
        # perform consensus clustering
        consensus_cluster_index = compute_spectral_clustering_consensus(self.clustering_assignments, self.n_clusters)
        # save clustering predictions computed by bagging step
        self.cluster_labels_ = consensus_cluster_index

        # update clustering matrix S
        S = self.predict_clusters_proba_from_cluster_labels(X)
        if self.negative_weighting in ['uniform']:
            S[index_negatives] = 1 / self.n_clusters
        elif self.negative_weighting in ['hard']:
            S[index_negatives] = np.rint(S[index_negatives])
        if self.positive_weighting in ['hard']:
            S[index_positives] = np.rint(S[index_positives])

        cluster_index = self.run_EM(X, y_polytope, S, consensus_cluster_index, index_positives, index_negatives, -1)

        # save barycenters and final predictions
        self.cluster_labels_ = cluster_index
        X_proj = X @ self.orthonormal_basis[-1].T
        self.barycenters = [
            np.mean(X_proj[index_positives][cluster_index == cluster], 0)[None, :] for cluster in
            range(np.max(cluster_index) + 1)]
