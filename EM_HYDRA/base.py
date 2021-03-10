from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod


class BaseEM(BaseEstimator, metaclass=ABCMeta):
    """Basic class for our Machine Learning Expectation-Maximization framework."""

    @abstractmethod
    def __init__(self, C, kernel, stability_threshold, noise_tolerance_threshold,
                 n_consensus, n_iterations, n_labels, n_clusters_per_label,
                 initialization, clustering, classification, consensus,
                 negative_weighting, positive_weighting):

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
            self.adaptive_clustering = True
        elif clustering in ['plug'] :
            self.n_clusters_per_label = n_clusters_per_label
            self.adaptive_clustering = True
        else:
            self.n_clusters_per_label = n_clusters_per_label
            self.adaptive_clustering = False

        # define what type of initialization, clustering, classification and consensus one wants to use
        self.initialization = initialization
        self.clustering = clustering
        self.classification = classification
        self.consensus = consensus

        # define what are the weightings we want for each label
        self.negative_weighting = negative_weighting
        self.positive_weighting = positive_weighting
