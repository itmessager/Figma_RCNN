


import numpy as np
import tensorflow as tf

males =tf.constant(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
#males =tf.constant(np.array([-2]))

male_logits = tf.constant(np.array([(0.84, 0.16), (0.72, 0.28), (0.86, 0.14), (0.97, 0.03), (0.91, 0.07),
                                    (0.9, 0.1), (0.72, 0.28), (0.84, 0.16), (0.97, 0.03), (0.99, 0.01)],dtype='float32'))


male_logits2 = tf.constant(np.array([(0, 1), (1, 1), (0, 0),(0, 1), (1, 0),
                                     (1, 0), (0, 1), (1, 0), (0, 1)],dtype='float32'))
# male_logits = tf.constant(np.array([(0.84, 0)],dtype='float32'))


def convert2D(logits):
    logits2D = tf.ones_like(logits) - logits
    return tf.concat([logits2D, logits], 1)




def all_correlation_cost(male_logits):
    """
    Args:
        :param attr_logits: n,
    Returns:
        label_loss, box_loss
    """

    # def correlation(name1, name2, f):
    #     return f(male_logits[name1], male_logits[name2])
    #
    # cor_cost = [correlation('male', 'skirt', f3),
    #             correlation('longsleeve', 'shorts', f3),
    #             correlation('formal', 'longpants', f2),
    #             correlation('tshirt', 'shorts', f1),
    #             correlation('longpants', 'skirt', f3),
    #             correlation('formal', 'shorts', f3)]
    # cor_cost = tf.add_n(cor_cost)
    cor_cost=f3(male_logits[:,0], male_logits[:,1], 10)
    return cor_cost


def f1(a1, a2, k):
    return tf.pow(a1 - a2, k)


def f2(a1, a2, k):
    return tf.exp(-k * (tf.pow(a1 - 1, 2) + tf.pow(a2, 2)))


def f3(a1, a2, k):
    return tf.exp(-k * (tf.pow(a1 - 1, 2) + tf.pow(a2 - 1, 2)))






def attr_losses(male_labels, male_logits):
    """
    Args:
        males: n,[-1,0,1,1,0]  int64
        male_logits: nx2 [(0.4,0.6),(0.72,0.28),(0.84,0.16),(0.17,0.83),(0.49,0.51)]  float64
    Returns:
        male_loss
    """

    valid_inds = tf.where(male_labels >= 0)
    valid_male_labels = tf.reshape(tf.gather(male_labels,valid_inds), [-1])
    valid_male_logits = tf.reshape(tf.gather(male_logits,valid_inds), (-1, 2))

    # valid_male_logits2D= convert2D(tf.gather(attribute_logits,valid_inds))

    AP = tf.metrics.average_precision_at_k(labels=valid_male_labels,predictions=valid_male_logits,k=1)[1]

    male_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_male_labels, logits=valid_male_logits)
    male_loss = tf.reduce_sum(male_loss) * (1. / 16.0)

    male_loss2 = tf.losses.sparse_softmax_cross_entropy(labels=valid_male_labels, logits=valid_male_logits)
    male_loss2 = tf.reduce_sum(male_loss2) * (1. / 16.0)
   # male_loss = tf.reduce_mean(male_loss, name='label_loss')

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):

        prediction = tf.argmax(valid_male_logits, axis=-1, name='label_prediction')

        # expend dim to prevent divide by zero
        #accuracy1 = tf.reduce_mean(correct, name='accuracy')

        accuracy2 = tf.metrics.mean_per_class_accuracy(labels=valid_male_labels, predictions=prediction, num_classes=2)[1]
        acc = tf.reduce_mean(accuracy2)
    return valid_male_labels, valid_male_logits, male_loss,prediction, male_loss,male_loss2,AP,acc



if __name__=='__main__':
    some_var = all_correlation_cost(male_logits2)
    sess = tf.Session()
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(init_op)  # initialize v
    sess.run(tf.local_variables_initializer())
    some_var = sess.run(some_var)


    print(some_var)






















