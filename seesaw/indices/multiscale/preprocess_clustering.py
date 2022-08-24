import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd

def preprocess_clusters(dataset, save_path): 
    vecs = dataset.vectors.to_numpy()
    #print("Creating Distances")
    #distance = pairwise_distances(vecs, metric='cosine')
    print("Making Scan")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    print("Clustering")
    #db = clusterer.fit(distance.astype('float64'))
    db = clusterer.fit(vecs)
    print("Saving")
    data = {}
    data['labels'] = db.labels_
    data['prob'] = db.probabilities_
    df = pd.DataFrame(data)
    df.to_parquet(save_path)
    
import argparse
from seesaw.definitions import resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset for use by Seesaw")
    parser.add_argument(
        "--multiscale_path",
        type=str,
        required=True,
        help="which dataset (list of files) to run on",
    )
    args = parser.parse_args()

    import ray
    from seesaw.dataset import SeesawDatasetManager
    from seesaw.indices.multiscale.preprocessor import preprocess_dataset, load_vecs

    #ray.init("auto", namespace="seesaw")

    print("Loading Vectors")
    df = load_vecs(args.multiscale_path)
    output_path = resolve_path(args.multiscale_path)
    preprocess_clusters(df, f"{output_path}/clusters.parquet")