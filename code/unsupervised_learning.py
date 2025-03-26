from data_preprocessing import query_to_dataset, data_files, queries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import data_visualization
import data_preprocessing
import numpy as np

data_file = "small_prolog"
query_name = "factors_all_weighted"
seed = 1234


def cluster_dataset(dataset, num_clusters, weights = None):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, init="random", random_state=seed)
    kmeans.fit(dataset, sample_weight=weights)
    return (kmeans)

def clustering_dim_reduced(dataset, num_clusters, weights = None):
    dataset_reduced = PCA(2).fit_transform(dataset)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, init="random", random_state=seed)
    kmeans.fit(dataset_reduced, sample_weight=weights)
    return kmeans, dataset_reduced
    

def elbow_test(dataset, max_clusters, weights = None):
    kmeans = KMeans(n_init=10, init="random")
    visualizer = KElbowVisualizer(kmeans, k = (1, max_clusters))
    dataset = np.array(dataset)
    visualizer.fit(dataset, sample_weight = weights)
    return visualizer



def visualize_clusters(num_clusters, dataset = None, weights = None):
    if dataset is None:
        dataset = query_to_dataset(data_files[data_file], queries[query_name])

    dataset_reduced = PCA(2).fit_transform(dataset)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, init="random", random_state=seed)
    labels = kmeans.fit_predict(dataset_reduced, sample_weight=weights)

    data_visualization.plot_clusters(dataset_reduced, labels, kmeans.cluster_centers_)


# Main for elbow method
def visualize_elbow():
    max_clusters = 20
    dataset = np.array(query_to_dataset(data_files[data_file], queries[query_name]))
    weight_index = 0
    weights = [np.log(data.pop(weight_index))/np.log(100) for data in dataset]
    visualizer = elbow_test(dataset, max_clusters, weights=weights)
    data_visualization.plot_elbow(visualizer)


# Main for clustering
def cluster_data(visualize = False):
    dataset = query_to_dataset(data_files[data_file], queries[query_name])
    weight_index = 0
    num_clusters = 4
    weights = [np.log(data.pop(weight_index))/np.log(100) for data in dataset]        
    dataset = np.array(dataset)

    kmeans = cluster_dataset(dataset, num_clusters, weights=weights)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    clusters = []
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        clusters.append(dataset[indices])
    

    if visualize:
        visualize_clusters(num_clusters, dataset, weights=weights)
        i = 0
        for centroid in centroids:
            img_name = f"cluster_center{i}"
            data_visualization.plot_factors(centroid, img_name)
            i += 1 
    
    return clusters

# Output data to new prolog clauses
def output_data(clusters):
    fact_head = data_preprocessing.facts_pieces["heads"]["person_clustered"]
    fact_tail = data_preprocessing.facts_pieces["tails"]["person_clustered"]
    facts = data_preprocessing.clusters_to_facts(clusters, fact_head, fact_tail)
    rules = data_preprocessing.rules["small_prolog_clustered"]
    clauses = rules + facts
    prolog_path = data_preprocessing.data_files["small_prolog_clustered"]
    data_preprocessing.write_prolog(clauses, prolog_path)
    return



def main():
    global seed
    seed = np.random.randint(100)
    print("[*] Clustering the data...")
    clusters = cluster_data(visualize=False)    
    print("[+] Data successfully clustered")
    print("[*] Converting data into facts and writing to prolog file...")
    output_data(clusters)
    print("[+] Data successfully written to prolog file")




if __name__ == "__main__":
    main()
