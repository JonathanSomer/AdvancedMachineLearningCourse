from itertools import combinations
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from sklearn.externals import joblib
from concurrent.futures import ProcessPoolExecutor

import data_utils as du
import numpy as np
import argparse
import requests
import config
import os


def update(msg):
    payload = {'message': msg, 'channel': config.slack_channel}
    requests.post(config.slack_url, json=payload)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def quadruplets_size(n_categories, n_clusters):
    return (n_categories * (n_categories - 1)) * (n_clusters * (n_clusters - 1))


def preprocess(n_files, silent=False):
    dataset_name = 'dataset_for_collection_nf.{0}'.format(n_files)
    read_path = du.read_pickle_path(dataset_name)
    if os.path.exists(read_path):
        if not silent:
            update('Loaded dataset from file')
        dataset = joblib.load(read_path)
    else:
        update('Fetching processed data from {0} {1}'.format(n_files, 'files' if n_files > 1 else 'file'))
        data_obj = du.get_processed_data(num_files_to_fetch_data_from=n_files)

        update('Creating low shot dataset')
        dataset = du.to_low_shot_dataset(data_obj)
        write_path = du.write_pickle_path(dataset_name)
        joblib.dump(dataset, write_path)
        update('done :tada: dataset for quadruplets collection saved as *{0}*'.format(dataset_name))

    cat_to_vectors, cat_to_onehots, original_shape = dataset
    return cat_to_vectors, cat_to_onehots, original_shape


def process_centroids(n_files, n_clusters, n_jobs, cat_to_vectors):
    centroids_name = 'centroids_nf.{0}_nc.{1}'.format(n_files, n_clusters)
    read_path = du.read_pickle_path(centroids_name)
    if os.path.exists(read_path):
        update('Loaded centroids from file')
        centroids = joblib.load(read_path)
    else:
        update('Creating clusters/centroids by {0} jobs'.format(n_jobs))
        centroids = {}
        for category, X in cat_to_vectors.items():
            update('Running KMeans to get {0} clusters/centroids for `{1}`.'.format(n_clusters, category))
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, verbose=1).fit(X)
            centroids[category] = kmeans.cluster_centers_

        write_path = du.write_pickle_path(centroids_name)
        joblib.dump(centroids, write_path)
        update('done to cluster :tada: centroids saved as *{0}*'.format(centroids_name))
    
    return centroids


def process_quadruplets_for_pair(n_files, n_clusters, a, b, centroids_a=None, centroids_b=None, silent=False):
    quadruplets_name = 'quadruplets/quadruplets_nf.{0}_nc.{1}_a.{2}_b.{3}'.format(n_files, n_clusters, a, b)
    read_path = du.read_pickle_path(quadruplets_name)
    if os.path.exists(read_path):
        if not silent:
            update('Loaded quadruplets for pair `{0}`, `{1}` from file'.format(a, b))
        quadruplets = joblib.load(read_path)
    else:
        quadruplets = []
        for c1a, c2a in combinations(centroids_a, 2):
            min_dist, pair_b = float('inf'), None
            for c1b, c2b in combinations(centroids_b, 2):
                dist = cosine(c1a - c2a, c1b - c2b)
                if dist < min_dist:
                    min_dist, pair_b = dist, (c1b, c2b)

                # maybe switching improves
                dist = cosine(c1a - c2b, c1b - c2a)
                if dist < min_dist:
                    min_dist, pair_b = dist, (c2b, c1b)

            c1b, c2b = pair_b
            if cosine_similarity(c1a - c2a, c1b - c2b) > 0:
                # quadruplet = (c1a, c2a, c1b, c2b)
                # quadruplets.append(quadruplet)

                # not sure if the second one is needed or not
                quadruplets.extend([(c1a, c2a, c1b, c2b),
                                    (c2a, c1a, c2b, c1b)])

        write_path = du.write_pickle_path(quadruplets_name)
        joblib.dump(quadruplets, write_path)
        update('done :tada: quadruplets for pair `{0}`, `{1}` saved as *{2}*'.format(a, b, quadruplets_name))

    return quadruplets


def load_quadruplets(n_clusters, categories, n_files=12):
    cat_to_vectors, cat_to_onehots, original_shape = preprocess(n_files=n_files, silent=True)
    centroids = process_centroids(n_files, n_clusters, 1, cat_to_vectors)
    quadruplets = []
    for a, b in combinations(categories, 2):
        quadruplets.extend(process_quadruplets_for_pair(n_files, n_clusters, a, b, centroids[a], centroids[b], silent=True))
        quadruplets.extend(process_quadruplets_for_pair(n_files, n_clusters, b, a, centroids[b], centroids[a], silent=True))

    return quadruplets, cat_to_vectors, cat_to_onehots, original_shape


def main(n_files, n_clusters, n_jobs, test, stop_instance):
    if test:
        n_files = 1
        n_clusters = 10

    update('*Generating low shot data procedure has just started* :weight_lifter:')

    cat_to_vectors, cat_to_onehots, original_shape = preprocess(n_files)
    centroids = process_centroids(n_files, n_clusters, n_jobs, cat_to_vectors)

    update('Creating quadruplets (2 pairs of 2 centroids).')
    with ProcessPoolExecutor() as executor:
        for a, b in combinations(centroids, 2):
            executor.submit(process_quadruplets_for_pair, n_files, n_clusters, a, b, centroids[a], centroids[b])
            executor.submit(process_quadruplets_for_pair, n_files, n_clusters, b, a, centroids[b], centroids[a])

    update('done to create `{0}` quadruplets :tada:'.format(quadruplets_size(len(centroids), n_clusters)))

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
