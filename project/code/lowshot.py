from itertools import combinations
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from sklearn.externals import joblib
from callbacks import CloudCallback

import data_utils as du
import numpy as np
import argparse
import requests
import config


def update(msg):
    payload = {'message': msg, 'channel': config.slack_channel}
    requests.post(config.slack_url, json=payload)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# creates lowshot data and pickle it.
def main(n_files, n_clusters, n_jobs, test, stop_instance):
    if test:
        n_files = 1
        n_clusters = 10

    update('*Generating low shot data procedure has just started* :weight_lifter:')
    update('Fetching processed data from {0} {1}'.format(n_files, 'files' if n_files > 1 else 'file'))
    data_obj = du.get_processed_data(num_files_to_fetch_data_from=n_files)
    update('done :tada:')

    update('Creating low shot dataset')
    dataset = du.to_low_shot_dataset(data_obj)
    cat_to_vectors, cat_to_onehots, original_shape = dataset
    update('done :tada:')

    update('Creating clusters by {0} jobs'.format(n_jobs))
    clusters = {}
    for category, X in cat_to_vectors.items():
        update('Running KMeans to get {0} clusters for "{1}".'.format(n_clusters, category))
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, verbose=1).fit(X)
        clusters[category] = kmeans
    update('done :tada:')

    update('Creating quadruplets (2 pairs of 2 centroids).')
    quadruplets = []
    for a, b in combinations(clusters, 2):
        for c1a, c2a in combinations(clusters[a].cluster_centers_, 2):
            min_dist, quadruplet, category = float('inf'), None, None
            for c1b, c2b in combinations(clusters[b].cluster_centers_, 2):
                dist = cosine(c1a - c2a, c1b - c2b)
                if dist < min_dist:
                    min_dist, quadruplet, category = dist, (c1a, c2a, c1b, c2b), a

            c1a, c2a, c1b, c2b = quadruplet
            if cosine_similarity(c1a - c2a, c1b - c2b) > 0:
                quadruplets.append((quadruplet, a))
    update('done :tada:')

    name = 'lowshot_f_{0}_c_{1}'.format(n_files, n_clusters)
    update('Saving data as *{0}*'.format(name))
    centroids = {category: cluster.cluster_centers_ for category, cluster in clusters.items()}

    data = {'clusters': clusters,
            'centroids': centroids,
            'dataset': dataset}

    file_path = du.read_pickle_path(name)
    joblib.dump(data, file_path)
    update('done :tada:')

    if stop_instance:
        update('Stopping instance')
        requests.get(config.stop_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=100)
    parser.add_argument('-j', '--n_jobs', help='number of jobs to do in parallel', type=int, default=8)
    parser.add_argument('-s', '--stop_instance', help='stop instance when run ends or not', action='store_true')
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')

    args = parser.parse_args()

    main(args.n_files, args.n_clusters, args.n_jobs, args.test, args.stop_instance)