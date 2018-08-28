from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from sklearn.externals import joblib
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder
from mnist_data import *

import data_utils as du
import numpy as np
import argparse
import requests
import config
import os


def to_low_shot_xray():
    data_obj = du.get_processed_data(num_files_to_fetch_data_from=12)
    return du.to_low_shot_dataset(data_obj)


dataset_to_n_categories = {'xray': 15,
                           'mnist': 10}

dataset_to_lowshot_func = {'xray': to_low_shot_xray,
                           'mnist': MnistData().to_low_shot_dataset}


def update(msg):
    print(msg)


def slack_update(msg):
    print(msg)
    payload = {'message': msg, 'channel': config.slack_channel}
    requests.post(config.slack_url, json=payload)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def quadruplets_size(n_categories, n_clusters):
    return (n_categories * (n_categories - 1)) * (n_clusters * (n_clusters - 1))


def preprocess(dataset_name, n_files, verbose=False):
    if dataset_name == 'xray':
        save_name = 'dataset_for_collection_nf.{0}'.format(n_files)
    else:
        save_name = '{0}_for_collection'.format(dataset_name)

    read_path = du.read_pickle_path(save_name)
    if os.path.exists(read_path):
        if verbose:
            update('Loaded dataset from file')
        return joblib.load(read_path)
    else:
        dataset = dataset_to_lowshot_func[dataset_name]()

    write_path = du.write_pickle_path(save_name)
    joblib.dump(dataset, write_path)
    update('done :tada: dataset for quadruplets data collection saved as *{0}*'.format(save_name))
    
    cat_to_vectors, original_shape = dataset
    return cat_to_vectors, original_shape


def process_centroids(dataset_name, n_files, n_clusters, cat_to_vectors, n_jobs, verbose=False, evaluate=False):
    if dataset_name == 'xray':
        centroids_name = 'centroids_nf.{0}_nc.{1}'.format(n_files, n_clusters)
    else:
        centroids_name = '{0}_centroids_{1}_clusters'.format(dataset_name, n_clusters)

    cat_to_inertia = {}

    read_path = du.read_pickle_path(centroids_name)
    if os.path.exists(read_path) and not evaluate:
        if verbose:
            update('Loaded centroids from file')
        cat_to_centroids = joblib.load(read_path)
    else:
        update('Creating clusters/centroids by {0} jobs'.format(n_jobs))
        cat_to_centroids = {}
        for category, X in cat_to_vectors.items():
            update('Running KMeans to get {0} clusters/centroids for `{1}`.'.format(n_clusters, category))
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, verbose=0).fit(X)
            cat_to_centroids[category] = kmeans.cluster_centers_
            cat_to_inertia[category] = kmeans.inertia_

        write_path = du.write_pickle_path(centroids_name)
        joblib.dump(cat_to_centroids, write_path)
        update('done to cluster :tada: centroids saved as *{0}*'.format(centroids_name))

    if evaluate:
        avg_inertia = sum(cat_to_inertia.values()) / len(cat_to_inertia)
        return avg_inertia

    return cat_to_centroids


def process_quadruplets_for_pair(dataset_name, n_files, n_clusters, a, b, centroids_a=None, centroids_b=None, verbose=False):
    if dataset_name == 'xray':
        quadruplets_name = 'quadruplets/quadruplets_nf.{0}_nc.{1}_a.{2}_b.{3}'.format(n_files, n_clusters, a, b)
    else:
        quadruplets_name = 'quadruplets/{3}_{0}.clusters_a.{1}_b.{2}'.format(n_clusters, a, b, dataset_name)
        
    read_path = du.read_pickle_path(quadruplets_name)
    if os.path.exists(read_path):
        if verbose:
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


def load_quadruplets(n_clusters, categories='all', n_files=12, dataset_name='xray'):
    cat_to_vectors, original_shape = preprocess(dataset_name, n_files)
    cat_to_centroids = process_centroids(dataset_name, n_files, n_clusters, cat_to_vectors, 1)

    if categories == 'all':
        categories = list(range(dataset_to_n_categories[dataset_name]))

    # filter unwanted categories
    cat_to_centroids = {category: cs for category, cs in cat_to_centroids.items() if category in categories}

    quadruplets = defaultdict(list)
    for a, b in combinations(categories, 2):
        quadruplets[a].extend(
            process_quadruplets_for_pair(dataset_name, n_files, n_clusters, a, b, cat_to_centroids[a], cat_to_centroids[b]))
        quadruplets[b].extend(
            process_quadruplets_for_pair(dataset_name, n_files, n_clusters, b, a, cat_to_centroids[b], cat_to_centroids[a]))

    return quadruplets, cat_to_centroids, cat_to_vectors, original_shape


def cross_validate(dataset_name='mnist', min_n=10, max_n=40, init_step=10, n_jobs=8, n_files=12):
    cat_to_vectors, original_shape = preprocess(dataset_name, n_files)

    best = max_n
    low, high, step = min_n, max_n, init_step
    while step > 0:
        inertias = {low: process_centroids(dataset_name, n_files, low, cat_to_vectors, n_jobs, evaluate=True),
                    high: process_centroids(dataset_name, n_files, low, cat_to_vectors, n_jobs, evaluate=True)}
        for n in range(low + step, high, step):
            inertias[n] = process_centroids(dataset_name, n_files, n, cat_to_vectors, n_jobs, evaluate=True)

        best = min(inertias, key=lambda n: inertias[n])
        slack_update('(*[n_clusters cross validation]* current best is _{0}_'.format(best))
        if best in (low, high):
            return best

        if inertias[low] < inertias[high]:
            high = best
        elif inertias[low] > inertias[high]:
            low = best
        else:
            slack_update('WEIRD (best == low == high) = {0}'.format(low))

        step //= 2

    return best


def main(dataset_name, n_files, n_clusters, n_jobs, test):
    if test:
        n_files = 1
        n_clusters = 5

    update('*Generating low shot data procedure has just started* :weight_lifter:')

    cat_to_vectors, original_shape = preprocess(dataset_name, n_files)
    cat_to_centroids = process_centroids(dataset_name, n_files, n_clusters, cat_to_vectors, n_jobs)

    update('Creating quadruplets (2 pairs of 2 centroids).')
    with ProcessPoolExecutor() as executor:
        for a, b in combinations(cat_to_centroids, 2):
            executor.submit(process_quadruplets_for_pair, dataset_name, n_files, n_clusters, a, b, cat_to_centroids[a], cat_to_centroids[b])
            executor.submit(process_quadruplets_for_pair, dataset_name, n_files, n_clusters, b, a, cat_to_centroids[b], cat_to_centroids[a])

    update('done to create `{0}` quadruplets :tada:'.format(quadruplets_size(len(cat_to_centroids), n_clusters)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='what dataset to use', default='mnist')  # xray is the second one
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=20)
    parser.add_argument('-j', '--n_jobs', help='number of jobs to do in parallel', type=int, default=8)
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')
    parser.add_argument('-cv', '--cross_validate', help='whether to run cross validation', action='store_true')
    parser.add_argument('--min_n', help='min number of clusters to cross validate on', type=int, default=10)
    parser.add_argument('--max_n', help='max number of clusters to cross validate on', type=int, default=40)
    parser.add_argument('--init_step', help='init step to use for cross validation', type=int, default=10)

    args = parser.parse_args()

    if args.cross_validate:
        cross_validate(args.dataset, args.min_n, args.max_n, args.init_step, args.n_jobs, args.n_files)
    else:
        main(args.dataset, args.n_files, args.n_clusters, args.n_jobs, args.test)
