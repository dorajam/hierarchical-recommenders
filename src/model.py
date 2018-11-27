from __future__ import unicode_literals, division, print_function
import tensorflow as tf


class AutoRec(object):

    def __init__(self, batch_size, epochs, eta, num_users, num_items, num_hidden):
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = eta
        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden = num_hidden

    def build_network(self, inputs):
        first_layer = layers.fully_connected(inputs, self.num_hidden, activation_fn=tf.identity)
        output_layer = layers.fully_connected(first_layer, self.num_items, activation_fn=tf.nn.sigmoid)
        return output_layer


class HierarchicalAutoRec(object):
    
    def __init__(self, batch_size, epochs, eta, num_users, num_items, num_hidden, num_tags, num_categories, reducer):
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = eta
        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden = num_hidden
        self.num_tags = num_tags
        self.num_categories = num_categories
        self.reducer = reducer
    
    def build_network(self, inputs, tag_level_labels, category_level_labels):
        first_layer = layers.fully_connected(inputs, self.num_hidden, activation_fn=tf.identity)
        final_item_predictions = layers.fully_connected(first_layer, self.num_items, activation_fn=tf.nn.sigmoid)
        
        tag_predictions, category_predictions = _reduce_rows(final_item_predictions, 
                                                             tag_level_labels, 
                                                             category_level_labels,
                                                             num_items=self.num_items,
                                                             num_tags=self.num_tags,
                                                             num_categories=self.num_categories,
                                                             reducer=self.reducer)
        
        return final_item_predictions, tag_predictions, category_predictions

    def _reduce_rows(self, final_layer, tag_layer, category_layer, reducer=tf.reduce_max):
        """
        final_layer:    (batch_size x num_items)
        tag_layer:      (num_items x num_tags)
        category_layer: (num_tags x num_categories)

        returns:
            reduced_tag_layer     : (batch_size x num_tags)
            reduced_category_layer: (batch_size x num_categories)
        """


        propagated_final_layer = tf.multiply(tf.reshape(tag_layer, [self.num_tags, self.num_items]), tf.expand_dims(final_layer,1))
        reduced_tag_layer = reducer(propagated_final_layer, -1)

        propagated_tag_layer = tf.multiply(tf.expand_dims(reduced_tag_layer,1), tf.reshape(category_layer,[self.num_categories, self.num_tags]))
        reduced_category_layer = reducer(propagated_tag_layer, -1)
        return reduced_tag_layer, reduced_category_layer

    def propragate_for_prediction(self, predicted_user_items, true_user_items, tag_labels, category_labels, reducer):
        """
        :predicted_user_itemss: user's predicted scores across all items
        :true_user_items:       user's binarized true item interactions
        returns
        tag_predictions, cat_predictions
        """
        predicted_user_itemss = tf.multiple(predicted_user_items, true_user_items)
        return _reduce_rows(predicted_user_items,
                            tag_labels,
                            cat_labels,
                            num_items=self.num_items,
                            num_tags=self.num_tags,
                            num_categories=self.num_categories,
                            reducer=reducer)
