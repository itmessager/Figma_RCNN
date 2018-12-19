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
#males =tf.constant(np.array([-1, 0, 1, 1, 0, -1, 0, 1, 1, 0]))
males =tf.constant(np.array([1, -1, -1, -1, -1, -1, -1, -1, -1, -1]))
male_logits = tf.constant(np.array([(0.4, 0.5), (0.72, 0.58), (0.16, 0.84), (0.77, 0.83), (0.51, 0.49),(0.4, 0.4), (0.72, 0.58), (0.84, 0.16), (0.77, 0.83), (0.49, 0.51)]))
#
# valid_inds = tf.where(males >= 0)
# valid_males_label = tf.gather(males,valid_inds)
# valide_male_logits = tf.gather(male_logits,valid_inds)
#
# sess = tf.Session()
#
# a = sess.run(valid_males_label)
# b = sess.run(valide_male_logits)
# print(a,b)

#
def male_losses(male_labels, male_logits):
    """
    Args:
        males: n,[-1,0,1,1,0]
        male_logits: nx2 [(0.4,0.6),(0.72,0.28),(0.84,0.16),(0.17,0.83),(0.49,0.51)]
    Returns:
        male_loss
    """
    expand_dim = tf.constant([1.0])
    valid_inds = tf.where(male_labels >= 0)
    valid_males_label = tf.reshape(tf.gather(male_labels,valid_inds), [-1])
    valid_male_logits = tf.reshape(tf.gather(male_logits,valid_inds), [-1, 2])
    valid_males_label_2D = tf.one_hot(valid_males_label, 2)
    valid_males_label_2D = tf.cast(valid_males_label_2D,tf.float64)

    male_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=valid_males_label_2D, logits=valid_male_logits)
    male_loss = tf.reduce_sum(male_loss) * (1. / 16.0)
   # male_loss = tf.reduce_mean(male_loss, name='label_loss')

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(valid_male_logits, axis=1, name='label_prediction')
        correct1 = tf.to_float(tf.equal(prediction, valid_males_label))  # boolean/integer gather is unavailable on GPU
        # expend dim to prevent divide by zero
        correct = tf.concat([correct1, expand_dim], 0)
        accuracy = tf.reduce_mean(correct, name='accuracy')
    return valid_inds, prediction, correct1,correct, male_loss,accuracy


if __name__=='__main__':
    loss = male_losses(males, male_logits)
    sess = tf.Session()
    loss = sess.run(loss)

    print(loss)
