import pandas as pd
import sklearn

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from config import Config


class MessageAssembler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.svd = TruncatedSVD(2)
        self.pca = PCA(2)
    
    
    def keyword_vector_clustering(self, data: pd.DataFrame):
        keyword_vectorizer = TfidfVectorizer()
        
        keyword_vectors = keyword_vectorizer.fit_transform(data['formatted_message'])
        
        keyword_vector_kmeans = KMeans(n_clusters=self.config.n_clusters)
        data['top_keywords_cluster_id'] = keyword_vector_kmeans.fit_predict(keyword_vectors)
        
        return keyword_vectorizer, keyword_vectors, keyword_vector_kmeans, data
    
    
    def message_vector_clustering(self, data: pd.DataFrame):
        message_vectors = data['message_vector'].tolist()

        message_vector_kmeans = KMeans(n_clusters=self.config.n_clusters)
        data['message_vector_cluster_id'] = message_vector_kmeans.fit_predict(message_vectors)
        
        return message_vectors, message_vector_kmeans, data

    
    def summarize_modeling_results(self, data: pd.DataFrame):
        # Clustering and saving results
        keyword_vectorizer, keywords_vectors, keyword_vector_kmeans, data = self.keyword_vector_clustering(data)
        message_vectors, message_vector_kmeans, data = self.message_vector_clustering(data)
        
        print('Saving clustering results ... ')
        data.to_csv(f'{self.config.clustering_results_dir}/clustered_messages.csv', index=False)
        
        print('Clustering results saved.')
        
        # Plot clusters based on top keywords
        print('Plotting clusters based on top {self.config.n_keywords} keywords ...')
        self.plot_clusters_based_on_top_keywords(keywords_vectors, data)
        
        print('Clusters based on top {self.config.n_keywords} keywords plotted')
        
        # Plot clusters based on message vector
        print('Plotting clusters based on message vector ...')
        self.plot_clusters_based_on_message_vector(message_vectors, data)
        
        print('Clusters based on message vector plotted')
        
        # Finding top keywords in each message-vector cluster
        print(f'Recording top {self.config.n_keywords} keywords in each message-vector cluster ...')
                
        terms = keyword_vectorizer.get_feature_names_out()
        ordered_message_vector_clustering_centroids = message_vector_kmeans.cluster_centers_.argsort()[:, ::-1]
        
        top_keywords_in_each_message_vector_cluster = []

        for i in range(ordered_message_vector_clustering_centroids.shape[0]):
            keywords = ', '.join([terms[ind] for ind in ordered_message_vector_clustering_centroids[i, :self.config.n_keywords]])
            top_keywords_in_each_message_vector_cluster.append(keywords)
        
        top_keywords_in_each_message_vector_cluster_df = pd.DataFrame({'message_vector_cluster_id': range(self.config.n_clusters),
            'top_keywords': top_keywords_in_each_message_vector_cluster})
            
        top_keywords_in_each_message_vector_cluster_df.to_csv(f'{self.config.clustering_results_dir}/top keywords in each message-vector cluster.csv', index=False)
        
        print(f'Top {self.config.n_keywords} keywords in each message-vector cluster recorded.')
        
    
    def plot_clusters_based_on_top_keywords(self, keywords_vectors: list, data: pd.DataFrame) -> None:
        feature_reducer = self.svd.fit_transform(keywords_vectors)

        plt.figure(figsize=(10, 7))
        
        for i in range(self.config.n_clusters):
            messages_in_this_cluster = feature_reducer[data['top_keywords_cluster_id'] == i]
            plt.scatter(messages_in_this_cluster[:, 0], messages_in_this_cluster[:, 1], label=f'Cluster {i}')

        plt.legend()
        
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title(f'Clusters based on top {self.config.n_keywords} keywords')
        
        plt.savefig(f'{self.config.clustering_results_dir}/Clusters based on {self.config.n_keywords} top keywords.png')
        
    
    def plot_clusters_based_on_message_vector(self, message_vectors: list, data: pd.DataFrame) -> None:
        feature_reducer = self.pca.fit_transform(message_vectors)

        # Plotting the clusters
        plt.figure(figsize=(10, 7))
        
        for i in range(self.config.n_clusters):
            messages_in_this_cluster = feature_reducer[data['message_vector_cluster_id'] == i]
            plt.scatter(messages_in_this_cluster[:, 0], messages_in_this_cluster[:, 1], label=f'Cluster {i}')

        plt.legend()
        
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('Clusters based on message vectors')
        
        plt.savefig(f'{self.config.clustering_results_dir}/Clusters based on message vector.png')
