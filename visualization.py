import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


class Visualization:

    @staticmethod
    def vis_elbow(alg_name, scaled_data):
        # initialize kmeans parameters
        kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "random_state": 1,
        }

        # create list to hold SSE values for each k
        sse = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(scaled_data)
            sse.append(kmeans.inertia_)

        # visualize results
        plt.plot(range(1, 10), sse)
        plt.xticks(range(1, 10))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title(f'{alg_name}: Elbow Plot to determine ideal cluster number')
        plt.show()

    @staticmethod
    def vis_clusters(alg_name, x, y, X_data, data_labels):
        '''
        Visualize how the data was clustered

        One image using all the data, and a single decision tree as the underlying model
        makes a prediction for ∼ 10,000 equally and rectangularly spaced points
        :param alg_name: str, algorithm (Agglomerative or Divisive)
        :param x: data used to make clusters
        :param y: array of predicted cluster identifier
        :param data_labels, cluster identifier for each row in dataframe
        :param X_data: dataframe
        :return: plot, clusters our model makes
        '''
        sns.scatterplot(x=x,
                        y=y,
                        data=X_data,
                        hue=data_labels,
                        palette="rainbow").set_title(f'{alg_name} Clustering: Paris Housing by {x} and {y}')
        plt.show()

    @staticmethod
    def boxplot(alg_name, x_name, y_names, data):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'{alg_name}: Cluster Statistics for squareMeters and price')
        # create chart in each subplot
        sns.boxplot(data=data, x=x_name, y=y_names[0], ax=ax1)
        sns.boxplot(data=data, x=x_name, y=y_names[1], ax=ax2)
        plt.show()
