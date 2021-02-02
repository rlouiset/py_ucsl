import numpy as np
from base import BaseML
from utils import *
from sinkornknopp import *
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score
import cvxpy as cp

class HYDRA(BaseML):
    """ Computes and stores the average and current value.
    """
    def __init__(self, C=1, n_consensus=5, n_iterations=5, n_clusters_per_label=None, training_label_mapping=None, initialization_type="DPP", kernel="linear",
                 tolerance=0.05, clustering_strategy='original', consensus='original', name="HYDRA"):
        super().__init__(name)
        if n_clusters_per_label is None:
            n_clusters_per_label = {0: 2, 1: 2}

        self.C = C
        self.n_consensus = n_consensus
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.initialization_type = initialization_type
        self.clustering_strategy = clustering_strategy
        self.consensus = consensus
        self.kernel = kernel

        self.training_label_mapping = training_label_mapping
        if training_label_mapping is not None :
            self.labels = np.unique(list(training_label_mapping.values()))
        else :
            self.labels = [0, 1]
        self.barycenters = {label:None for label in self.labels}
        self.n_clusters_per_label = n_clusters_per_label

        self.coefficients = {label:{cluster_i:None for cluster_i in range(n_clusters_per_label[label])} for label in self.labels}
        self.intercepts = {label:{cluster_i:None for cluster_i in range(n_clusters_per_label[label])} for label in self.labels}

        self.S_lists = {label:dict() for label in self.labels}
        self.coef_lists = {label:{cluster_i:dict() for cluster_i in range(n_clusters_per_label[label])} for label in self.labels}
        self.intercept_lists = {label:{cluster_i:dict() for cluster_i in range(n_clusters_per_label[label])} for label in self.labels}

        self.mean_direction={label:None for label in self.labels}
        self.mean_intercept = {label:0 for label in self.labels}
        self.intercept_bank = 0

    def fit(self, X_train, y_train):
        if self.training_label_mapping is not None :
            for original_label, new_label in self.training_label_mapping.items() :
                y_train[y_train==original_label] = new_label

        for label in self.labels :
            self.run(X_train, y_train, idx_outside_polytope=label)

    def predict(self, X):
        if len(self.labels) == 2 :
            y_pred = self.predict_binary_proba(X)[:,1]
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0
        else :
            y_pred = self.predict_proba(X)
            y_pred = np.argmax(y_pred, 1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict_binary_proba(self, X):
        SVM_scores_dict = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in self.labels}
        y_pred = np.zeros((len(X), 2))
        for label in self.labels:
            ## fullfill the SVM score matrix
            for cluster_i in range(self.n_clusters_per_label[label]):
                SVM_coefficient, SVM_intercept = self.coefficients[label][cluster_i], self.intercepts[label][cluster_i]
                SVM_scores_dict[label][:, cluster_i] = (np.matmul(SVM_coefficient, X.transpose()) + SVM_intercept).transpose().squeeze()

        ## fullfill each cluster score
        for i in range(len(X)):
            y_pred[i][1] = sigmoid(np.max(SVM_scores_dict[1][i, :]) - np.max(SVM_scores_dict[0][i, :]))
            y_pred[i][0] = 1-y_pred[i][1]
        return y_pred

    def predict_proba(self, X):
        ''' '''
        ## predict cluster distance for each label
        if len(self.labels) == 2 :
            y_pred_proba = self.predict_binary_proba(X)
        else :
            distance_predictions = self.predict_SVM_distances(X)
            y_pred_proba = np.zeros((len(X), len(distance_predictions)))
            for sample in range(len(X)) :
                for label_i in self.labels :
                    y_pred_proba[sample][label_i] = distance_predictions[label_i][sample, 0]
        return y_pred_proba


    def predict_distances(self, X):
        cluster_predictions = {label: np.zeros((len(X), self.n_clusters_per_label[label] + 1)) for label in self.labels}

        if self.clustering_strategy in ['original'] :
            SVM_scores_dict = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in self.labels}

            for label in self.labels:
                ## fullfill the SVM score matrix
                for cluster_i in range(self.n_clusters_per_label[label]):
                    SVM_coefficient, SVM_intercept = self.coefficients[label][cluster_i], self.intercepts[label][cluster_i]
                    SVM_scores_dict[label][:, cluster_i] = 1+(np.matmul(SVM_coefficient, X.transpose()) + SVM_intercept).transpose().squeeze()

                ## fullfill each cluster score
                for i in range(len(X)):
                    if np.max(SVM_scores_dict[label][i, :]) <= 0:
                        cluster_predictions[label][i, 0] = sigmoid(np.mean(SVM_scores_dict[label][i, :]))  # P(y=label)
                    else:
                        cluster_distance_vect = SVM_scores_dict[label][i, :]
                        cluster_predictions[label][i, 0] = sigmoid(np.sum(cluster_distance_vect[cluster_distance_vect > 0]))

                    for cluster_i in range(self.n_clusters_per_label[label]):
                        SVM_scores_dict[label][i, cluster_i] = max(SVM_scores_dict[label][i, cluster_i], 0)            # sigmoid(SVM_scores_dict[label][i, cluster_i])
                    for cluster_i in range(self.n_clusters_per_label[label]):
                        cluster_predictions[label][i, cluster_i + 1] = SVM_scores_dict[label][i, cluster_i] / (np.sum(SVM_scores_dict[label][i, :])+0.000001)                                      # P(cluster=i|y=label)
            # norm_column = np.sum(np.concatenate([(cluster_predictions[label][:,0])[:,None] for label in self.labels], axis=1), 1)
            # for label in self.labels:
            #    cluster_predictions[label][:,0] /= norm_column

        elif self.clustering_strategy in ['boundary_barycenter'] :
            barycenters_scores_dict = {label : np.zeros((len(X), self.n_clusters_per_label[label])) for label in self.labels}
            for label in self.labels:
                for cluster_i in range(self.n_clusters_per_label[label]) :
                    w_cluster_i = self.coefficients[label][cluster_i]
                    b_cluster_i = self.intercepts[label][cluster_i]
                    w_cluster_i_norm = w_cluster_i / np.linalg.norm(w_cluster_i)**2
                    boundary_barycenter_i = self.barycenters[label][cluster_i] + (self.barycenters[label][cluster_i]@w_cluster_i[0]+b_cluster_i)*w_cluster_i_norm
                    barycenters_scores_dict[label][:,cluster_i] = sigmoid(-np.linalg.norm(X-boundary_barycenter_i, axis=1))
                for cluster_i in range(self.n_clusters_per_label[label]):
                    cluster_predictions[label][:, cluster_i+1] = barycenters_scores_dict[label][:, cluster_i] / np.sum(barycenters_scores_dict[label], 1)     # P(cluster=i|y=label)

        elif self.clustering_strategy in ['mean_hp']:
            cluster_predictions = {label: np.zeros((len(X), self.n_clusters_per_label[label] + 1)) for label in self.labels}
            mean_hp_scores = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in self.labels}
            for label in self.labels:
                X_proj = (np.matmul(self.mean_direction[label][None,:], X.transpose()) + self.mean_intercept[label]).transpose().squeeze()
                X_proj = sigmoid(X_proj[:, None]*5/np.max(X_proj))

                cluster_predictions[label][:, 1] = (1-X_proj)[:,0]
                cluster_predictions[label][:, 2] = X_proj[:,0]

        if self.consensus in ['mean_hp']:
            cluster_predictions = {label: np.zeros((len(X), self.n_clusters_per_label[label] + 1)) for label in self.labels}
            for label in self.labels:
                X_proj = (np.matmul(self.mean_direction[label][None, :], X.transpose()) + self.mean_intercept[label]).transpose().squeeze()
                X_proj = sigmoid(X_proj * 5 / np.max(X_proj))

                cluster_predictions[label][:, 1] = (1 - X_proj)
                cluster_predictions[label][:, 2] = X_proj

        return cluster_predictions


    def predict_cluster_assignement(self, X):
        cluster_predictions = self.predict_distances(X)
        for key in cluster_predictions.keys() :
            cluster_predictions[key] = cluster_predictions[key][:,1:]
        return cluster_predictions

    def run(self, X, y, idx_outside_polytope):
        n_clusters = self.n_clusters_per_label[idx_outside_polytope]
        n_consensus = self.n_consensus if (n_clusters > 1) else 1
        ## put the label idx_center_polytope at the center of the polytope by setting it to positive labels
        y_polytope = np.copy(y)
        y_polytope[y_polytope!=idx_outside_polytope] = -1    ## if label is inside of the polytope, the distance is negative and the label is not divided into
        y_polytope[y_polytope==idx_outside_polytope] = 1     ## if label is outside of the polytope, the distance is positive and the label is clustered

        consensus_assignment = np.zeros((len(y_polytope), n_consensus))
        consensus_direction = []
        consensus_intercepts = []

        index_positives = np.where(y_polytope == 1)[0]  # index for Positive Labels
        index_negatives = np.where(y_polytope == -1)[0]  # index for Negative Labels

        for consensus_i in range(n_consensus):
            ## depending on the weight initialization strategy, random hyperplanes were initialized with maximum diversity to constitute the convex polytope
            S, cluster_index = self.init_S(X, y_polytope, index_positives, index_negatives, n_clusters, idx_outside_polytope, initialization_type=self.initialization_type)
            self.S_lists[idx_outside_polytope][0]=S.copy()

            for cluster_i in range(n_clusters):
                cluster_i_weight = np.ascontiguousarray(S[:, cluster_i])
                SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_i_weight, kernel=self.kernel)
                self.coefficients[idx_outside_polytope][cluster_i] = SVM_coefficient
                self.intercepts[idx_outside_polytope][cluster_i] = SVM_intercept

                self.coef_lists[idx_outside_polytope][cluster_i][0] = SVM_coefficient.copy()
                self.intercept_lists[idx_outside_polytope][cluster_i][0] = SVM_intercept.copy()

            for iter in range(self.n_iterations):
                ## decide the convergence of the polytope based on the toleration
                S_hold = S.copy()
                S, cluster_index = self.update_S(X, y, S, index_positives, cluster_index, idx_outside_polytope)
                self.S_lists[idx_outside_polytope][1+iter]=S.copy()

                if self.clustering_strategy in ['original', 'nw_mean_hp']:
                    S[index_negatives, :] = 1/n_clusters
                if self.consensus in ['original', 'w_original']:
                    S[index_negatives, :] = 1/n_clusters
                S[index_positives, :] = 0
                S[index_positives, cluster_index[index_positives]] = 1

                ## update barycenters
                label_barycenters = np.zeros((S.shape[1], X.shape[1]))
                for cluster_i in range(n_clusters):
                    label_barycenters[cluster_i] = np.mean(X[index_positives] * S[index_positives, cluster_i][:, None],0)
                self.barycenters[idx_outside_polytope] = label_barycenters

                ## check the loss comparted to the tolorence for stopping criteria
                cluster_consistency = ARI(np.argmax(S[index_positives],1), np.argmax(S_hold[index_positives],1))
                if cluster_consistency > 0.8 :
                    break

                for cluster_i in range(n_clusters):
                    if np.count_nonzero(S[index_positives, cluster_i]) == 0 :
                        #print("Cluster dropped, meaning that all Positive Labels has been assigned to one single hyperplane in iteration: %d" % ( iter - 1))
                        print("Re-initialization of the clustering...")
                        S, cluster_index = self.init_S(X, y_polytope, index_positives, index_negatives, n_clusters, idx_outside_polytope, initialization_type=self.initialization_type)

                for cluster_i in range(n_clusters):
                    cluster_i_weight = np.ascontiguousarray(S[:, cluster_i])
                    SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_i_weight, kernel=self.kernel)
                    self.coefficients[idx_outside_polytope][cluster_i] = SVM_coefficient
                    self.intercepts[idx_outside_polytope][cluster_i] = SVM_intercept

                    self.coef_lists[idx_outside_polytope][cluster_i][iter+1] = SVM_coefficient.copy()
                    self.intercept_lists[idx_outside_polytope][cluster_i][iter+1] = SVM_intercept.copy()

            ## update the cluster index for the consensus clustering
            consensus_assignment[:, consensus_i] = cluster_index
            consensus_direction.append([self.coefficients[idx_outside_polytope][cluster_i][0] for cluster_i in range(len(self.coefficients[idx_outside_polytope]))])
            consensus_intercepts.append(self.intercept_bank)

        if n_consensus > 1 :
            self.apply_consensus(X, y_polytope, consensus_assignment, consensus_direction, consensus_intercepts, n_clusters, index_positives, index_negatives, idx_outside_polytope)

    def update_S(self, X, y, S, index, cluster_index, idx_outside_polytope, lambda_sigmoid=5) :
        if self.n_clusters_per_label[idx_outside_polytope] == 1 :
            S[index] = 1
            cluster_index[index] = 0
            return S, cluster_index

        Q = S.copy()
        if self.clustering_strategy == 'original':
            svm_scores = np.zeros(S.shape)
            for cluster_i in range(self.n_clusters_per_label[idx_outside_polytope]) :
                 ## Apply the data again the trained model to get the final SVM scores
                 svm_scores[:, cluster_i] = 1+(np.matmul(self.coefficients[idx_outside_polytope][cluster_i], X.transpose()) + self.intercepts[idx_outside_polytope][cluster_i]).transpose().squeeze()
            Q = py_softmax(svm_scores, 1)


        elif self.clustering_strategy in ['mean_hp', 'nw_mean_hp']:
            directions = np.array([self.coefficients[idx_outside_polytope][cluster_i][0] for cluster_i in range(self.n_clusters_per_label[idx_outside_polytope])])
            intercepts = np.array([self.intercepts[idx_outside_polytope][cluster_i][0] for cluster_i in range(self.n_clusters_per_label[idx_outside_polytope])])

            ###
            X_0 = (np.matmul(directions[0][None,:], X.transpose()) + intercepts[0]).transpose().squeeze()
            X_1 = (np.matmul(directions[1][None,:], X.transpose()) + intercepts[1]).transpose().squeeze()
            min_indices = np.argpartition(np.abs(X_0)+np.abs(X_1), 10)
            ###

            directions[0] = directions[0]*np.linalg.norm(directions[1])**2 / np.mean((np.linalg.norm(directions, axis=1)**2))
            directions[1] = directions[1]*np.linalg.norm(directions[0])**2 / np.mean((np.linalg.norm(directions, axis=1)**2))
            mean_direction = (directions[0] - directions[1])/2

            ###
            mean_intercept = - np.mean(X[min_indices]@mean_direction)
            self.intercept_bank = mean_intercept
            ###

            X_norm = X.copy()
            X_proj = (np.matmul(mean_direction[None,:], X_norm.transpose()) + mean_intercept).transpose().squeeze()
            X_proj = sigmoid(X_proj[:, None]*lambda_sigmoid/np.max(X_proj))

            Q = np.concatenate((1 - X_proj, X_proj), axis=1)

        S = Q.copy()
        cluster_index[index] = np.argmax(Q[index], axis=1)
        return S, cluster_index


    def init_S(self, X, y_polytope, index_positives, index_negatives, n_clusters, idx_outside_polytope, initialization_type="DPP") :
        if n_clusters==1 :
            S = np.ones((len(y_polytope), n_clusters)) / n_clusters
            cluster_index = np.argmax(S, axis=1)
            self.barycenters[idx_outside_polytope] = np.mean(X, 0)[None, 1]
            return S, cluster_index

        S = np.ones((len(y_polytope), n_clusters)) / n_clusters
        weight_samples = np.zeros((len(index_positives), S.shape[1]))

        if initialization_type == "DPP":  ##
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
            prob = np.zeros((len(X), n_clusters))  # only consider the PTs

            for i in range(n_clusters):
                prob[:, i] = np.matmul(
                    np.multiply(X, np.divide(1, np.linalg.norm(X, axis=1))[:, np.newaxis]),
                    W[Widx[i], :].transpose())

            l = np.minimum(prob - 1, 0)
            d = prob - 1
            weight_samples = proportional_assign(l, d)

        S = weight_samples.copy()  ## only replace the sample weight for positive samples
        cluster_index = np.argmax(S, axis=1)

        return S, cluster_index

    def launch_svc(self, X, y, sample_weight, kernel) :
        SVC_clsf = SVC(kernel=kernel, C=self.C)
        ## fit the different SVM/hyperplanes
        SVC_clsf.fit(X, y, sample_weight=sample_weight)

        SVM_coefficient = SVC_clsf.coef_
        SVM_intercept = SVC_clsf.intercept_

        return SVM_coefficient, SVM_intercept

    def apply_consensus(self, X, y_polytope, consensus_assignment, consensus_direction, consensus_intercepts, n_clusters, index_positives,
                        index_negatives, idx_outside_polytope):
        S = np.ones((len(y_polytope), n_clusters)) / n_clusters
        if self.consensus == 'original':
            ## do censensus clustering
            consensus_scores = consensus_clustering(consensus_assignment[index_positives].astype(int), n_clusters)
            ## change the weight of positivess to be 1, negatives to be 1/_clusters
            # then set the positives' weight to be 1 for the assigned hyperplane
            S[index_positives, :] *= 0
            S[index_positives, consensus_scores] = 1

        if self.consensus == 'w_original':
            ## clustering relevancy
            w_clusterings = np.zeros((consensus_assignment.shape[1], consensus_assignment.shape[1]))
            for clustering_i in range(len(w_clusterings)) :
                for clustering_j in range(len(w_clusterings)) :
                    if clustering_j != clustering_i :
                        w_clusterings[clustering_i][clustering_j] = ARI(consensus_assignment[:,clustering_i], consensus_assignment[:,clustering_j])
            w_clusterings = np.sum(w_clusterings, 1)
            w_clusterings[w_clusterings<0] = 0
            w_clusterings = w_clusterings / np.sum(w_clusterings)

            ## do censensus clustering
            consensus_scores = consensus_clustering(consensus_assignment[index_positives].astype(int), n_clusters, cluster_weight=w_clusterings)
            ## change the weight of positivess to be 1, negatives to be 1/_clusters
            # then set the positives' weight to be 1 for the assigned hyperplane
            S[index_positives, :] *= 0
            S[index_positives, consensus_scores] = 1

        if self.consensus == 'neg_w_original':
            ## clustering relevancy
            w_clusterings = np.zeros((consensus_assignment.shape[1], consensus_assignment.shape[1]))
            for clustering_i in range(len(w_clusterings)) :
                for clustering_j in range(len(w_clusterings)) :
                    if clustering_j != clustering_i :
                        w_clusterings[clustering_i][clustering_j] = ARI(consensus_assignment[:,clustering_i], consensus_assignment[:,clustering_j])
            w_clusterings = np.sum(w_clusterings, 1)
            w_clusterings[w_clusterings<0] = 0
            w_clusterings = w_clusterings / np.sum(w_clusterings)

            ## do censensus clustering
            consensus_scores = consensus_clustering_neg(consensus_assignment.astype(int), n_clusters, index_positives, cluster_weight=w_clusterings)
            ## change the weight of positivess to be 1, negatives to be 1/_clusters
            # then set the positives' weight to be 1 for the assigned hyperplane
            S = consensus_scores.copy()
            S[index_positives, :] *= 0
            S[index_positives, np.argmax(consensus_scores[index_positives], 1)] = 1


        elif self.consensus == 'mean_hp':
            ## clustering relevancy
            w_clusterings = np.zeros((consensus_assignment.shape[1], consensus_assignment.shape[1]))
            for clustering_i in range(len(w_clusterings)) :
                for clustering_j in range(len(w_clusterings)) :
                    if clustering_j != clustering_i :
                        w_clusterings[clustering_i][clustering_j] = ARI(consensus_assignment[:,clustering_i], consensus_assignment[:,clustering_j])
            w_clusterings = np.sum(w_clusterings, 1)
            w_clusterings[w_clusterings<0] = 0
            w_clusterings = w_clusterings / np.sum(w_clusterings)
            ## do censensus clustering
            y_clustering_positives = consensus_clustering(consensus_assignment.astype(int), n_clusters, cluster_weight=w_clusterings)
            X_positives = X[index_positives]

            max_ARI = 0
            for consensus_i in range(self.n_consensus) :
                directions_i = consensus_direction[consensus_i]
                intercept_i = consensus_intercepts[consensus_i]

                directions_i = directions_i / (np.linalg.norm(directions_i, axis=1)**2)[:, None]

                mean_direction_i = directions_i[0,:]-directions_i[1,:]
                mean_direction_i /= 2

                distances_positives_i = X_positives @ mean_direction_i + intercept_i
                pred_positives_i = np.rint(sigmoid(distances_positives_i)).astype(np.int)

                ARI_i = ARI(pred_positives_i, y_clustering_positives)

                if ARI_i > max_ARI :
                    max_ARI = ARI_i
                    self.mean_direction[idx_outside_polytope] = mean_direction_i
                    self.mean_intercept[idx_outside_polytope] = intercept_i

            X_proj = X@self.mean_direction[idx_outside_polytope] + self.mean_intercept[idx_outside_polytope]
            X_proj = sigmoid(X_proj * 5 / np.max(X_proj))

            S = np.concatenate(((1-X_proj)[:,None], X_proj[:,None]), axis=1)

            # then set the positives' weight to be 1 for the assigned hyperplane
            S[index_positives, :] *= 0
            S[index_positives, np.rint(X_proj[index_positives]).astype(np.int)] = 1

            self.S_lists[idx_outside_polytope][-1] = S.copy()


        for cluster_i in range(n_clusters):
            cluster_weight = np.ascontiguousarray(S[:, cluster_i])
            SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_weight, self.kernel)
            self.coefficients[idx_outside_polytope][cluster_i] = SVM_coefficient
            self.intercepts[idx_outside_polytope][cluster_i] = SVM_intercept

            self.coef_lists[idx_outside_polytope][cluster_i][-1] = SVM_coefficient.copy()
            self.intercept_lists[idx_outside_polytope][cluster_i][-1] = SVM_intercept.copy()

        if self.consensus in ['original', 'w_original', 'neg_w_original'] :
            directions = np.array([self.coefficients[idx_outside_polytope][cluster_i][0] for cluster_i in range(self.n_clusters_per_label[idx_outside_polytope])])
            intercepts = np.array([self.intercepts[idx_outside_polytope][cluster_i][0] for cluster_i in range(self.n_clusters_per_label[idx_outside_polytope])])

            X_0 = (np.matmul(directions[0][None,:], X.transpose()) + intercepts[0]).transpose().squeeze()
            X_1 = (np.matmul(directions[1][None,:], X.transpose()) + intercepts[1]).transpose().squeeze()
            min_indices = np.argpartition(np.abs(X_0)+np.abs(X_1), 10)

            directions[0] = directions[0]*np.linalg.norm(directions[1])**2 / np.mean((np.linalg.norm(directions, axis=1)**2))
            directions[1] = directions[1]*np.linalg.norm(directions[0])**2 / np.mean((np.linalg.norm(directions, axis=1)**2))
            self.mean_direction[idx_outside_polytope] = (directions[0] - directions[1])/2
            self.mean_intercept[idx_outside_polytope] = - np.mean(X[min_indices]@self.mean_direction[idx_outside_polytope])