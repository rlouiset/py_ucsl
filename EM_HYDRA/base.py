from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod


class BaseEM(BaseEstimator, metaclass=ABCMeta):
    """Basic class for our Machine Learning Expectation-Maximization framework."""

    @abstractmethod
    def __init__(self, stability_threshold, noise_tolerance_threshold,
                 n_consensus, n_iterations,
                 initialization, clustering, classification, consensus,
                 negative_weighting, positive_weighting,
                 custom_clustering_method, custom_classification_method, custom_initialization_matrixes):

        # define hyperparameters such as stability_threshold or noise_tolerance_threshold
        assert (0 < stability_threshold <= 1), "The stability_threshold value is invalid. It must be between 0 and 1."
        assert (noise_tolerance_threshold > 1), "The noise_tolerance_threshold value is invalid. It must be at least 1."
        self.stability_threshold = stability_threshold
        self.noise_tolerance_threshold = noise_tolerance_threshold

        # define number of iterations or consensus to perform
        assert (type(n_iterations) is int and n_iterations > 1), "The number of iterations must be an integer, at least 2"
        assert (type(n_consensus) is int and n_consensus > 0), "The number of consensus must be an integer, at least 1"
        self.n_consensus = n_consensus
        self.n_iterations = n_iterations

        # define what type of initialization, clustering, classification and consensus one wants to use
        assert (initialization in ['k_means', 'gaussian_mixture', 'DPP', 'precomputed']), \
            "Initialization must be one of 'k_means', 'gaussian_mixture', 'DPP', 'precomputed'"
        if initialization == 'custom' :
            assert (custom_initialization_matrixes is not None), \
                "if initialization is custom you have to pass a initialization_matrixes different from None"
            self.custom_initialization_matrixes = custom_initialization_matrixes
        self.initialization = initialization

        assert (clustering in ['k_means', 'gaussian_mixture', 'HYDRA', 'custom']), \
            "Clustering must be one of 'k_means', 'gaussian_mixture', 'HYDRA', 'custom'"
        if clustering == 'custom' :
            assert (custom_clustering_method is not None), \
                "if clustering is custom you have to pass a custom_clustering_method different from None"
            self.custom_clustering_method = custom_clustering_method
        self.clustering = clustering

        assert (classification in ['max_margin', 'logistic', 'custom']), \
            "Classification must be one of 'max_margin', 'logistic', 'custom'"
        if classification == 'custom' :
            assert (custom_classification_method is not None), \
                "if classification is custom you have to pass a custom_classification_method different from None"
            self.custom_classification_method = custom_classification_method
        self.classification = classification

        assert (consensus in ['spectral_clustering']), \
            "Classification must be one of 'spectral_clustering'"
        self.consensus = consensus

        # define what are the weightings we want for each label
        assert (negative_weighting in ['hard_clustering', 'soft_clustering', 'all']), \
            "negative_weighting must be one of 'hard_clustering', 'soft_clustering'"
        assert (positive_weighting in ['hard_clustering', 'soft_clustering']), \
            "positive_weighting must be one of 'hard_clustering', 'soft_clustering'"
        self.negative_weighting = negative_weighting
        self.positive_weighting = positive_weighting

