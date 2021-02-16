from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta, abstractmethod

from sklearn.metrics import adjusted_rand_score as ARI
from EM_HYDRA.sinkornknopp import *
from EM_HYDRA.DPP_utils import *
from EM_HYDRA.utils import *
from sklearn.svm import SVC


class BaseEM(BaseEstimator, metaclass=ABCMeta):
    """Basic class for our Machine Learning Expectation-Maximization framework."""

    @abstractmethod
    def __init__(self, C, kernel, stability_threshold, noise_tolerance_threshold,
                 n_consensus, n_iterations, n_labels, n_clusters_per_label,
                 initialization, clustering, consensus, negative_weighting, dual_consensus):

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
        self.dual_consensus = dual_consensus


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

    def __init__(self, C=1, kernel="linear", stability_threshold=0.9, noise_tolerance_threshold=5,
                 n_consensus=5, n_iterations=5, n_labels=2, n_clusters_per_label=None,
                 initialization="DPP", clustering='original', consensus='spectral_clustering', negative_weighting='all',
                 training_label_mapping=None, dual_consensus=False):

        super().__init__(C=C, kernel=kernel, stability_threshold=stability_threshold,
                         noise_tolerance_threshold=noise_tolerance_threshold,
                         n_consensus=n_consensus, n_iterations=n_iterations, n_labels=n_labels,
                         n_clusters_per_label=n_clusters_per_label,
                         initialization=initialization, clustering=clustering, consensus=consensus,
                         negative_weighting=negative_weighting, dual_consensus=dual_consensus)

        # define the mapping of labels before fitting the algorithm
        # for example, one may want to merge 2 labels together before fitting to check if clustering separate them well
        if training_label_mapping is None:
            self.training_label_mapping = {label: label for label in range(self.n_labels)}
        else:
            self.training_label_mapping = training_label_mapping

        # define clustering parameters
        self.barycenters = {label: None for label in range(self.n_labels)}
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
        self.orthonormal_basis = {label: None for label in range(self.n_labels)}
        self.k_means = {label: None for label in range(self.n_labels)}
        self.gaussian_mixture = {label: None for label in range(self.n_labels)}

    def fit(self, X_train, y_train):
        """Fit the HYDRA model according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self
        """
        # apply label mapping (in our case we merged "BIPOLAR" and "SCHIZOPHRENIA" into "MENTAL DISEASE" for our xp)
        for original_label, new_label in self.training_label_mapping.items():
            y_train[y_train == original_label] = new_label

        # cluster each label one by one and confine the other inside the polytope
        for label in range(self.n_labels):
            self.run(X_train, y_train, self.n_clusters_per_label[label], idx_outside_polytope=label)

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
            Predictions of the labels of the query points
        """
        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, 1)

    def predict_proba(self, X):
        y_pred = np.zeros((len(X), 2))

        # first compute points distances to hyperplane
        SVM_distances = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in range(self.n_labels)}

        for label in range(self.n_labels):
            # fullfill the SVM distances \w the original HYDRA formulation
            for cluster_i in range(self.n_clusters_per_label[label]):
                SVM_coefficient = self.coefficients[label][cluster_i]
                SVM_intercept = self.intercepts[label][cluster_i]
                SVM_distances[label][:, cluster_i] = X @ SVM_coefficient[0] + SVM_intercept[0]

        if self.clustering in ['original']:
            # merge each label distances and compute the probability \w sigmoid function
            for i in range(len(X)):
                y_pred[i][1] = sigmoid(np.max(SVM_distances[1][i, :]) - np.max(SVM_distances[0][i, :]))
                y_pred[i][0] = 1 - y_pred[i][1]

        else:
            cluster_predictions = self.predict_clusters(X)
            # compute the predictions \w.r.t cluster previously found
            for i in range(len(X)):
                y_pred[i, 1] = sum([cluster_predictions[1][i, cluster] * SVM_distances[1][i, cluster] for cluster in
                                    range(self.n_clusters_per_label[1])])
                y_pred[i, 1] -= sum([cluster_predictions[0][i, cluster] * SVM_distances[0][i, cluster] for cluster in
                                     range(self.n_clusters_per_label[0])])
            # compute probabilities \w sigmoid
            y_pred[:, 1] = sigmoid(y_pred[:, 1] / np.max(y_pred[:, 1]))
            y_pred[:, 0] = 1 - y_pred[:, 1]

        return y_pred

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
                    SVM_distances[label][:, cluster] = X @ SVM_coefficient[0] + SVM_intercept[0]

                # compute clustering conditional probabilities as in the original HYDRA paper : P(cluster=i|y=label)
                SVM_distances[label] -= np.min(SVM_distances[label])
                SVM_distances[label] += 1e-3
                SVM_distances[label] = SVM_distances[label] / np.sum(SVM_distances[label], 1)[:, None]

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

        elif self.clustering in ['k_means']:
            for label in range(self.n_labels):
                X_proj = X @ self.orthonormal_basis[label].T
                y_proj_pred = self.k_means[label].predict(X_proj)

                cluster_predictions[label] = one_hot_encode(y_proj_pred, n_classes=self.n_clusters_per_label[label])

        elif self.clustering in ['gaussian_mixture']:
            for label in range(self.n_labels):
                X_proj = X @ self.orthonormal_basis[label].T
                y_proj_pred = self.gaussian_mixture[label].predict(X_proj)

                cluster_predictions[label] = one_hot_encode(y_proj_pred, n_classes=self.n_clusters_per_label[label])

        print(cluster_predictions)

        return cluster_predictions

    def run(self, X, y, n_clusters, idx_outside_polytope):
        n_consensus = self.n_consensus if (self.n_clusters_per_label[idx_outside_polytope] > 1) else 1

        # set label idx_outside_polytope outside of the polytope by setting it to positive labels
        y_polytope = np.copy(y)
        # if label is inside of the polytope, the distance is negative and the label is not divided into
        y_polytope[y_polytope != idx_outside_polytope] = -1
        # if label is outside of the polytope, the distance is positive and the label is clustered
        y_polytope[y_polytope == idx_outside_polytope] = 1

        consensus_assignment = np.zeros((len(y_polytope), n_consensus))

        index_positives = np.where(y_polytope == 1)[0]  # index for Positive labels (outside polytope)
        index_negatives = np.where(y_polytope == -1)[0]  # index for Negative labels (inside polytope)

        for consensus_i in range(n_consensus):
            # first we initialize the clustering matrix S, with the initialization strategy set in self.initialization
            S, cluster_index = self.initialize_clustering(X, y_polytope, index_positives, index_negatives,
                                                          n_clusters, idx_outside_polytope)

            # TODO : Get rid of these visualization helps
            self.S_lists[idx_outside_polytope][0] = S.copy()

            for cluster in range(n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_assignment)
                self.coefficients[idx_outside_polytope][cluster] = SVM_coefficient
                self.intercepts[idx_outside_polytope][cluster] = SVM_intercept

                # TODO: get rid of
                self.coef_lists[idx_outside_polytope][cluster][0] = SVM_coefficient.copy()
                self.intercept_lists[idx_outside_polytope][cluster][0] = SVM_intercept.copy()

            for iteration in range(self.n_iterations):
                # decide the convergence based on the clustering stability
                S_hold = S.copy()
                S, cluster_index = self.update_clustering(X, S, index_positives, cluster_index, n_clusters,
                                                          idx_outside_polytope)

                # TODO : get rid of
                self.S_lists[idx_outside_polytope][iteration + 1] = S.copy()

                # applying the negative weighting set as input
                if self.negative_weighting == 'all':
                    S[index_negatives] = 1 / n_clusters
                elif self.negative_weighting == 'hard_clustering':
                    S[index_negatives] = np.rint(S[index_negatives]).astype(np.float)

                # always set positive clustering as hard
                S[index_positives] = 0
                S[index_positives, cluster_index[index_positives]] = 1

                # check the Clustering Stability \w Adjusted Rand Index for stopping criteria
                cluster_consistency = ARI(np.argmax(S[index_positives], 1), np.argmax(S_hold[index_positives], 1))
                print(cluster_consistency)

                if cluster_consistency > self.stability_threshold:
                    break

                # check for degenerate clustering for positive labels (warning) and negatives (might be normal)
                for cluster in range(self.n_clusters_per_label[idx_outside_polytope]):
                    if np.count_nonzero(S[index_positives, cluster]) == 0:
                        print(
                            "Cluster dropped, meaning that one cluster have no positive points anymore, in iteration: %d" % (
                                    iteration - 1))
                        print("Re-initialization of the clustering...")
                        S, cluster_index = self.initialize_clustering(X, y_polytope, index_positives, index_negatives,
                                                                      n_clusters, idx_outside_polytope)

                    if np.count_nonzero(S[index_negatives, cluster]) == 0:
                        print(
                            "Cluster too far, meaning that one cluster have no negative points anymore, in iteration: %d" % (
                                    iteration - 1))
                        print("Re-distribution of this cluster negative weight to 'all...'")
                        S[index_negatives, cluster] = 1 / n_clusters

                for cluster in range(n_clusters):
                    cluster_assignment = np.ascontiguousarray(S[:, cluster])
                    SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_assignment)
                    self.coefficients[idx_outside_polytope][cluster] = SVM_coefficient
                    self.intercepts[idx_outside_polytope][cluster] = SVM_intercept

                    # TODO: get rid of
                    self.coef_lists[idx_outside_polytope][cluster][iteration + 1] = SVM_coefficient.copy()
                    self.intercept_lists[idx_outside_polytope][cluster][iteration + 1] = SVM_intercept.copy()
            print('')

            # update the cluster index for the consensus clustering
            consensus_assignment[:, consensus_i] = cluster_index + 1

        if n_consensus > 1:
            self.clustering_bagging(X, y_polytope, consensus_assignment, n_clusters, index_positives, idx_outside_polytope)

        return self

    def update_clustering(self, X, S, index, cluster_index, n_clusters, idx_outside_polytope):
        if n_clusters == 1:
            S[index] = 1
            cluster_index[index] = 0
            return S, cluster_index

        Q = S.copy()
        if self.clustering == 'original':
            SVM_distances = np.zeros(S.shape)
            for cluster in range(self.n_clusters_per_label[idx_outside_polytope]):
                # Apply the data again the trained model to get the final SVM scores
                SVM_distances[:, cluster] = 1 + (
                        np.matmul(self.coefficients[idx_outside_polytope][cluster], X.transpose()) +
                        self.intercepts[idx_outside_polytope][cluster]).transpose().squeeze()
            SVM_distances -= np.min(SVM_distances)
            SVM_distances += 1e-3
            Q = SVM_distances / np.sum(SVM_distances, 1)[:,None]

        if self.clustering in ['k_means', 'gaussian_mixture']:
            directions = [self.coefficients[idx_outside_polytope][cluster_i][0] for cluster_i in
                          range(self.n_clusters_per_label[idx_outside_polytope])]

            basis = []
            for v in directions:
                w = v - np.sum(np.dot(v, b) * b for b in basis)
                if np.linalg.norm(w) * self.noise_tolerance_threshold > 1:
                    basis.append(w / np.linalg.norm(w))

            self.orthonormal_basis[idx_outside_polytope] = np.array(basis)
            X_proj = X @ self.orthonormal_basis[idx_outside_polytope].T

            centroids = [np.mean(S[index, cluster_i][:, None] * X_proj[index, :], 0) for cluster_i in
                         range(self.n_clusters_per_label[idx_outside_polytope])]

            if self.clustering == 'k_means':
                self.k_means[idx_outside_polytope] = KMeans(n_clusters=self.n_clusters_per_label[idx_outside_polytope],
                                                            init=np.array(centroids), n_init=1).fit(X_proj[index])
                Q = one_hot_encode(self.k_means[idx_outside_polytope].predict(X_proj),
                                   n_classes=self.n_clusters_per_label[idx_outside_polytope])

            if self.clustering == 'gaussian_mixture':
                self.gaussian_mixture[idx_outside_polytope] = GaussianMixture(
                    n_components=self.n_clusters_per_label[idx_outside_polytope]).fit(X_proj[index])
                Q = self.gaussian_mixture[idx_outside_polytope].predict_proba(X_proj)

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

        S = Q.copy()
        cluster_index[index] = np.argmax(Q[index], axis=1)
        return S, cluster_index

    def initialize_clustering(self, X, y_polytope, index_positives, index_negatives, n_clusters, idx_outside_polytope):
        if n_clusters == 1:
            S = np.ones((len(y_polytope), n_clusters)) / n_clusters
            cluster_index = np.argmax(S, axis=1)
            self.barycenters[idx_outside_polytope] = np.mean(X, 0)[None, 1]
            return S, cluster_index

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
                    np.multiply(X[index_positives, :], np.divide(1, np.linalg.norm(X[index_positives, :], axis=1))[:, np.newaxis]),
                    W[Widx[i], :].transpose())

            l = np.minimum(prob - 1, 0)
            d = prob - 1
            S[index_positives] = proportional_assign(l, d)

        if self.initialization == "k_means":
            KM = KMeans(n_clusters=self.n_clusters_per_label[idx_outside_polytope]).fit(X[index_positives])
            S = one_hot_encode(KM.predict(X))

        cluster_index = np.argmax(S, axis=1)
        return S, cluster_index

    def launch_svc(self, X, y, sample_weight):
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
        # fit the different SVM/hyperplanes
        SVM_classifier = SVC(kernel=self.kernel, C=self.C)
        SVM_classifier.fit(X, y, sample_weight=sample_weight)

        # get SVM intercept value
        SVM_intercept = SVM_classifier.intercept_

        # get SVM hyperplane coefficient
        if self.kernel == 'rbf':
            X_support = X[SVM_classifier.support_]
            y_support = y[SVM_classifier.support_]
            SVM_coefficient = SVM_classifier.dual_coef_ @ np.einsum('i,ij->ij', y_support, X_support)
        else:
            SVM_coefficient = SVM_classifier.coef_

        return SVM_coefficient, SVM_intercept

    def clustering_bagging(self, X, y_polytope, consensus_assignment, n_clusters, index_positives,
                           idx_outside_polytope):
        """Fit the classification SVMs according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        consensus_assignment : array-like, shape (n_samples, n_consensus)
            Clustering predicted for each consensus run.
        n_clusters : int
            number of clusters to be set.
        index_positives : array-like, shape (n_positives_samples,)
            indexes of the positive labels being clustered
        idx_outside_polytope : int
            label that is being clustered
        Returns
        -------
        None
        """
        # initialize the consensus clustering vector
        S = np.zeros(index_positives.shape)

        if self.consensus == 'spectral_clustering':
            # perform consensus clustering
            S = consensus_clustering(consensus_assignment.astype(int), n_clusters, index_positives,
                                     negative_weighting=self.negative_weighting)

        if self.consensus == 'weighted_spectral_clustering':
            # compute clustering relevancy to weight the spectral clustering
            clustering_weights = np.zeros((consensus_assignment.shape[1], consensus_assignment.shape[1]))
            for clustering_i in range(len(clustering_weights)):
                for clustering_j in range(len(clustering_weights)):
                    if clustering_j != clustering_i:
                        clustering_weights[clustering_i][clustering_j] = ARI(
                            consensus_assignment[index_positives, clustering_i],
                            consensus_assignment[index_positives, clustering_j])
            clustering_weights = np.sum(clustering_weights, 1)
            clustering_weights[clustering_weights < 0] = 0
            clustering_weights = clustering_weights / np.sum(clustering_weights)

            # do consensus clustering
            S = consensus_clustering(consensus_assignment, n_clusters, index_positives,
                                     negative_weighting=self.negative_weighting,
                                     cluster_weight=clustering_weights)

        for cluster in range(n_clusters):
            cluster_weight = np.ascontiguousarray(S[:, cluster])
            SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_weight)
            self.coefficients[idx_outside_polytope][cluster] = SVM_coefficient
            self.intercepts[idx_outside_polytope][cluster] = SVM_intercept

            # TODO: get rid of
            self.coef_lists[idx_outside_polytope][cluster][-1] = SVM_coefficient.copy()
            self.intercept_lists[idx_outside_polytope][cluster][-1] = SVM_intercept.copy()

        # update clustering one last time for methods such as k_means or bisector_hyperplane
        _, _ = self.update_clustering(X, S, index_positives, np.argmax(S, 1), n_clusters, idx_outside_polytope)

