# a = [0,2].append([1])
# print(a)
# list = [1,2,3]
# list.append(4)
# print(list)
# list2 = [5,6]
#
# print(type(list)==type([]))
# print(type(list))

import numpy as np
import tensorflow as tf

#males =tf.constant(np.array([-1, 1, 1, 1, -1, 1, -1, -1, 1, 1]))
males =tf.constant(np.array([-1, 1, -1, 1, 0, -1, 1, 0, 1, -1]))
male_logits = tf.constant(np.array([(0.44, 0.51), (0.72, 0.58), (0.16, 0.84), (0.77, 0.83), (0.51, 0.49),
                                    (0.4, 0.4), (0.72, 0.58), (0.84, 0.16), (0.77, 0.83), (0.49, 0.51)]
                                  ,dtype='float32'))
def attr_losses(male_labels, male_logits):
    """
    Args:
        males: n,[-1,0,1,1,0]  int64
        male_logits: nx2 [(0.4,0.6),(0.72,0.28),(0.84,0.16),(0.17,0.83),(0.49,0.51)]  float64
    Returns:
        male_loss
    """
    valid_inds = tf.where(male_labels >= 0)
    specific_labels = tf.where(male_labels >= 0, tf.ones_like(male_labels), tf.zeros_like(male_labels))
    specific_logits = tf.reshape(male_logits[:, 0], [-1])
    attribute_logits = male_logits[:, 1]
    specific_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(specific_labels), logits=specific_logits)
    specific_loss_mean = tf.reduce_mean(specific_loss)

    valid_male_labels = tf.reshape(tf.gather(male_labels,valid_inds), [-1])
    valid_male_logits = tf.reshape(tf.gather(attribute_logits,valid_inds), [-1])

    male_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(valid_male_labels), logits=valid_male_logits)
    male_loss = tf.reduce_sum(male_loss) * (1. / 16.0)
   # male_loss = tf.reduce_mean(male_loss, name='label_loss')

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        #prediction = tf.argmax(valid_male_logits, axis=1, name='label_prediction')
        prediction = tf.where(attribute_logits > 0.5, tf.ones_like(attribute_logits),tf.zeros_like(attribute_logits))
        prediction = tf.where(specific_logits < 0.5, -tf.ones_like(prediction), prediction)
        prediction = tf.to_int64(prediction, name='label_prediction')
        # positive_label = tf.where(valid_males_label == 1.0)

        correct = tf.to_float(tf.equal(prediction, male_labels))  # boolean/integer gather is unavailable on GPU
        # expend dim to prevent divide by zero
        accuracy = tf.reduce_mean(correct, name='accuracy')

    return valid_inds, prediction, male_labels, male_loss,specific_loss_mean


if __name__=='__main__':
    some_var = attr_losses(males, male_logits)
    sess = tf.Session()
    some_var = sess.run(some_var)

    print(some_var)
