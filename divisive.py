from sklearn.cluster import BisectingKMeans
# This line of code imports BisectingKMeans model from sk-learn
# The BisectingKMeans is an iterative variant of KMeans, using divisive hierarchical clustering

class DivisiveAlgorithm:

    def __init__(self, n_clusters=5):
        """
        Constructor sets up a sci-kit learn "AgglomerativeClustering" under the hood,
        which does most of the heavy lifting for us.

        :param n_clusters: number of clusters. The default here is 2
        """
        self.Divisive_hc = BisectingKMeans(n_clusters = n_clusters, random_state=0)

    def fit(self, X):
        """
        This is the method which will be called to train the model. We can assume that train will
        only be called one time for the purposes of this project.

        :param X: The samples and features which will be used for training. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]

        :return: self Think of this method not having a return statement at all. The idea to
        "return self" is a convention of scikit learn; the underlying model should have some
        internally saved trained state.
        """
        # Only one line of code is needed for this method.
        self.Divisive_hc = self.Divisive_hc.fit(X)
        return self

    def fit_predict(self, X):
        """
        This is the method which will be used to predict the output targets/responses of a given
        list of samples.

        It should rely on mechanisms saved after train(X, y) was called.
        You can assume that train(X, y) has already been called before this method is invoked for
        the purposes of this project.

        :param X array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        Training instances to cluster, or distances between instances if affinity='precomputed'.
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :return: Cluster labels.
        """
        # Only one line of code is needed for this method, too
        return self.Divisive_hc.fit_predict(X)
