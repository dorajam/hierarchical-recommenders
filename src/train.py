import tensorflow as tf
import numpy as np
import argparse

from utils import *
from model import AutoRec, HierarchicalAutoRec


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Choose from 'baseline', 'max_hrec', or 'avg_hrec'.", choices=["baseline", "max_hrec", "avg_hrec"])
    parser.add_argument("-b", "--batch_size",
                        help="Number of users that to be processed in one batch, default=128", default=128, type=int)
    parser.add_argument("-e", "--epochs",
                        help="Number of epochs to run training for, default=20.", default=20, type=int)
    parser.add_argument("-eta", "--eta", help="Learning Rate, default=0.001", default=0.001, type=float)
    parser.add_argument("-i_lambda", "--item_lambda",
                        help="Convex combination parameter on the item level of the taxonomy.", default=0.5, type=float)
    parser.add_argument("-t_lambda", "--tag_lambda",
                        help="Convex combination parameter on the tag level of the taxonomy.", default=0.3, type=float)
    parser.add_argument("-c_lambda", "--cat_lambda",
                        help="Convex combination parameter on the category level of the taxonomy.", default=0.2, type=float)
    parser.add_argument("-o", "--optimizer",
            help="Tensorflow optimizer")
    parser.add_argument("-k", "--k_folds",
                        help="Number of split datasets to read in.", default=1, type=int)

    args = parser.parse_args()
    model = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    eta = args.eta
    item_lambda = args.item_lambda
    tag_lambda = args.tag_lambda
    cat_lambda = args.cat_lambda
    optimizer = args.optimizer
    k = args.k_folds

    print "Loading MovieLense 20ML training and test sets..."

    # returns a list of [ui_train, taxonomy_train, ui_test, taxonomy_test]
    k_folds = read_k_folds(data_dir='../data/splits/', k=k)
    for [train, taxonomy_train, test, taxonomy_test] in k_folds:

        user_items_tr, items_to_tags_tr, tags_to_categories_tr = change_grain(train, taxonomy_train)
        user_items_ts, items_to_tags_ts, tags_to_categories_ts = change_grain(test, taxonomy_test)

        # print diagnostics
        users = list(set(user_items.index))
        items = list(set(items_to_tags.index))
        tags = list(set(tags_to_categories.index))
        categories = list(set(taxonomy_labels['category'].unique()))

        num_users = len(users)
        num_items = len(items)
        num_tags = len(tags)
        num_categories = len(categories)

        print '(users x rated items):', num_users, 'x', num_items
        print '(num of items)', num_items
        print '(num of tags)', num_tags
        print '(num of categories)', num_categories

        binarized_user_items_tr = binarize_dataset(user_items_tr, classes=items)
        binarized_items_to_tags_tr = binarize_dataset(items_to_tags_tr, classes=tags)
        binarized_tags_to_categories_tr = binarize_dataset(tags_to_categories_tr, classes=categories)
        
        binarized_user_items_ts = binarize_dataset(user_items_ts, classes=items)
        binarized_items_to_tags_ts = binarize_dataset(items_to_tags_ts, classes=tags)
        binarized_tags_to_categories_ts = binarize_dataset(tags_to_categories_ts, classes=categories)

        if model == 'baseline':
            autorec = AutoRec(batch_size=batch_size,
                              epochs=epochs,
                              eta=eta,
                              num_users=num_users,
                              num_items=num_items,
                              num_hidden=num_hidden)

        if model == 'max_hrec':
            autorec = HierarchicalAutoRec(batch_size=batch_size,
                                          epochs=epochs,
                                          eta=eta,
                                          num_users=num_users,
                                          num_items=num_items,
                                          num_hidden=num_hidden,
                                          num_tags=num_tags,
                                          num_categories=num_categories,
                                          reducer=tf.reduce_max)

            inputs = tf.placeholder(tf.float32, shape=(None, num_apps))
            outputs = ae.build_network(inputs)

            loss = tf.reduce_mean(tf.square(outputs - inputs))
            train_op = tf.train.AdamOptimizer(learning_rate=ae.lr).minimize(loss)

            batch_per_ep = ae.num_users // ae.batch_size
                    
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(ae.epoch_num):
                    data = shuffle_users(train_x.astype(np.float32))
                    for batch_n in range(batch_per_ep):
                        batch = data[batch_n * ae.batch_size: (batch_n + 1) * ae.batch_size]
                        
                        _, c = sess.run([train_op, loss], feed_dict={inputs: batch})

                    print('Epoch: {} - cost= {:.5f}'.format((epoch + 1), c))
                predictions = sess.run(outputs, feed_dict={inputs: data})

        get_precision(predictions, train_x.astype(np.float32), groundtruth_x.astype(np.float32))
        get_taxonomy_level_mrr(predictions, train_x, groundtruth_x, k=5)

        if model == 'avg_hrec':
            autorec = HierarchicalAutoRec(batch_size=batch_size,
                                          epochs=epochs,
                                          eta=eta,
                                          num_users=num_users,
                                          num_items=num_items,
                                          num_hidden=num_hidden,
                                          num_tags=num_tags,
                                          num_categories=num_categories,
                                          reducer=tf.reduce_mean)
