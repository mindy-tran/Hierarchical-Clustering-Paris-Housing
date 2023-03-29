import pandas as pd
from divisive import DivisiveAlgorithm
from visualization import Visualization
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


class Experiment02:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        X_data = Experiment02._load_data()

        # remove outliers
        X_data = Experiment02.remove_outliers(X_data)

        # normalization before computing distances
        X_data = Experiment02.normalize(X_data)

        # Elbow Method to Find the Optimal Number of Clusters
        Visualization.vis_elbow('Divisive', X_data)
        # we see that about 5 clusters are good

        # hierarchical clustering model training on the data
        agglo_clustering = DivisiveAlgorithm()
        # give each point a cluster number
        y_hc = agglo_clustering.fit_predict(X_data)
        data_labels = agglo_clustering.Divisive_hc.labels_

        # for ease of plotting, make into a pandas df
        X_data = pd.DataFrame(X_data, columns=('squareMeters', 'price'))
        X_data['Cluster'] = data_labels

        # visualize clusters
        Visualization.vis_clusters('Divisive', 'squareMeters', 'price', X_data, 'Cluster')

        # boxplot to evaluate the ML technique
        Visualization.boxplot('Divisive', 'Cluster', ['squareMeters', 'price'], X_data)

        # Evaluation metrics: silhouette_score, calinski_harabasz_score
        silhouette = silhouette_score(X_data, data_labels, metric='euclidean')
        calinski_harabasz = calinski_harabasz_score(X_data, data_labels)
        print(f'Silhouette score: {silhouette:.4}\nCalinski-Harabsz score: {calinski_harabasz:.10}')

    @staticmethod
    def _load_data(filename="cleaned_data.csv"):
        """
        Load the data, separating it into a list of samples and their corresponding outputs
        :param filename: The location of the data to load from file.
        :return: X; each as an iterable object(like a list or a numpy array).
        The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]

        y
        """
        # all but index column
        X = pd.read_csv(filename).iloc[:, 1:]

        return X

    @staticmethod
    def remove_outliers(features_data):
        '''
        Helper function.
        Hierarchical clustering is sensitive to outliers.
        Uses an interquartile range approach to remove points outside the quartiles +/-1.5 * interquartile range.

        :param features_data:
        :return: features_data but without outliers
        '''
        features_data_copy = features_data.copy()

        for col in list(features_data_copy.columns):
            # identify quartiles
            Q1 = features_data_copy[str(col)].quantile(0.05)
            Q3 = features_data_copy[str(col)].quantile(0.95)
            # interquartile range
            IQR = Q3 - Q1
            # +/-1.5 * interquartile range
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # remove outliers
            features_data_copy = features_data_copy[(features_data_copy[str(col)] >= lower_bound) &
                                                    (features_data_copy[str(col)] <= upper_bound)]

        return features_data_copy

    @staticmethod
    def normalize(no_outliers):
        '''
        Hierarchical clustering is very sensitive to different scales.
        We rescale all the variables before computing the distance.

        :param no_outliers: dataframe
        :return: Array: Normalized data
        '''
        features_data_scaler = StandardScaler()
        return features_data_scaler.fit_transform(no_outliers)


if __name__ == "__main__":
    # Run the experiment once.
    Experiment02.run()
