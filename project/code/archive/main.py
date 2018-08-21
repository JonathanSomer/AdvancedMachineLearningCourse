import os
import argparse

from config import *

from callbacks import CloudCallback


def datasets_paths(complete):
    if complete:
        raise NotImplementedError('Handling the complete datset was not implemented yet.')
    else:
        train_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'small', 'train')
        test_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'small', 'test')

    return os.path.abspath(train_path), os.path.abspath(test_path)


def pickle_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


def model_path(subpath):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', subpath)
    return os.path.abspath(path)


def main(disease, n_epochs, remote, complete):
    try:
        train_path, test_path = datasets_paths(complete)
        cb = CloudCallback(remote=remote, slack_url=slack_url, stop_url=stop_url, slack_channel=slack_channel)

        raise NotImplementedError('main was not implemented')

    except Exception as e:
        cb.send_update(repr(e))

        import traceback
        traceback.print_exc()
    finally:
        cb.stop_instance()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('disease', help='the low-shot disease to experiment on')
    parser.add_argument('-e', '--n_epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-r', '--remote', help='run remotely on the configured gcloud vm', action='store_true')
    # parser.add_argument('-a', '--diseases', help='get all available diseases', action='store_true')
    parser.add_argument('-c', '--complete', help='run on the complete dataset (and not only on small part of it', action='store_true')

    args = parser.parse_args()

    main(args.disease, args.n_epochs, args.remote, args.complete)
