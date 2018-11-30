import os
import shutil
import argparse

from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("k", help="Number of split datasets, default=1", default=1, type=int)
    parser.add_argument("-c", "--clean", help="Clean directory", default=True, type=bool)

    args = parser.parse_args()
    k = args.k
    clean = args.clean

    print("Loading MovieLense 20ML...")
    ratings, tags, movies = load_movielense_20m()
    taxonomy_labels = get_taxonomy_labels(tags, movies)
    ratings, taxonomy_labels = reduce_to_common_items(ratings, taxonomy_labels)

    if clean:
        if os.path.exists('../data/splits'):
            print "Cleaning directory..."
            shutil.rmtree('../data/splits')

    try:
        ratios = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9], k, replace=False)
    except:
        raise Exception('k is larger than the number of available split options.')

    for i in range(k):
        print 'Splitting fold {}...'.format(i)

        outdir = '../data/splits/' + str(i)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # ui_train, ui_validation, ui_test = three_way_split(ratings, ratio)

        ui_train, ui_test = two_way_split(ratings, ratios[i])

        ui_train.to_csv(os.path.join(outdir, 'ui_train.csv'))
        ui_test.to_csv(os.path.join(outdir, 'ui_test.csv'))
        taxonomy_labels.to_csv(os.path.join(outdir, 'taxonomy_labels.csv'))
