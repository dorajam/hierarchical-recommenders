import os
import numpy as np
import pandas as pd


def load_movielense_20m(folder='../data/ml-20m'):
    ratings = pd.read_csv(os.path.join(folder, 'ratings.csv'))
    tags = pd.read_csv(os.path.join(folder, 'tags.csv'))
    movies = pd.read_csv(os.path.join(folder, 'movies.csv'))

    filtered_ratings = _prepare_for_implicit_representations(ratings)
    tags = tags.drop(columns='timestamp')

    users = len(filtered_ratings['userId'].unique())
    rated_movies = len(filtered_ratings['movieId'].unique())
    print '(users x rated movies):', users, rated_movies

    return filtered_ratings, tags, movies


def three_way_split(df, ratio=(.7, .1, .2), timestamp_col='timestamp'):
    """
    Performs three-way time-based split to generate three datasets.
    
    :df:            User-item interactions dataset, one row per user, item, interaction timestamp
    :ratio:         Split ratio for train, validation, test set, respectively.
    :timestamp_col: Timestamp column name
    
    :rtype:         Validation, train, and test sets
    """
    train_frac, val_frac, test_frac = ratio
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([timestamp_col])
    
    anchor_train = int(df.shape[0] * train_frac)
    anchor_val = int(df.shape[0] * (val_frac+train_frac))
    
    train = df.iloc[:anchor_train]
    validation = df.iloc[anchor_train:anchor_val]
    test = df.iloc[anchor_val:]
    
    return train, validation, test


def get_taxonomy_labels(tags, movies):
    """
    Takes tags and movies dataframe, and returns a joined dataframe with
    one row per movieId, tag, category.
    """
    rows = []
    _ = movies.apply(lambda row: [rows.append([row['movieId'], row['title'], nn]) 
        for nn in row['genres'].split('|')] , axis=1)

    df = pd.DataFrame(rows, columns=movies.columns).set_index(['movieId', 'title'])
    movie_categories = df.reset_index()
    taxonomy_labels = pd.merge(tags, movie_categories, 'inner', 'movieId')[['tag', 'genres', 'movieId']].drop_duplicates()
    taxonomy_labels['category'] = taxonomy_labels['genres']
    return taxonomy_labels.drop('genres')


def generate_k_fold_split_datasets(original, timestamp_col='timestamp', k=3):
    k_folds = []
    for _ in xrange(k):
        train_frac = np.random.choice(map(lambda r: r/10., range(3,10)))
        val_frac = np.random.choice(map(lambda r: r/10., range(1, int((1.0 - train_frac) * 10))))
        test_frac = 1.0 - train_frac
        train, validation, test = three_way_split(original, (train_frac, val_frac, test_frac), timestamp_col)
        k_folds.append([train, validation, test])

    return k_folds


def _prepare_for_implicit_representations(ratings):
    return ratings[ratings.rating >= 3.5]

