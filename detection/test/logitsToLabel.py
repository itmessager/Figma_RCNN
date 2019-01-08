import numpy as np
import tensorflow as tf

attr_logits = tf.constant(np.array([(0.44, 0.51), (0.72, 0.58), (0.16, 0.84), (0.77, 0.83), (0.51, 0.49),
                                    (0.4, 0.65), (0.72, 0.58), (0.84, 0.16), (0.77, 0.83), (0.49, 0.51),
                                    (0.4, 0.4), (0.72, 0.58), (0.44, 0.16), (0.57, 0.83), (0.79, 0.51)]
                                   , dtype='float32'))
def logits_to_label(attr_logits):
    """
    Args:
        males: n,[-1,0,1,1,0]  int64
        male_logits: nx2 [(0.4,0.6),(0.72,0.28),(0.84,0.16),(0.17,0.83),(0.49,0.51)]  float64
    Returns:
        male_loss
    """
    specific_logits = attr_logits[:, 0]
    attribute_logits = attr_logits[:, 1]


    #prediction = tf.argmax(valid_male_logits, axis=1, name='label_prediction')
    prediction = tf.where(attribute_logits > 0.5, tf.ones_like(attribute_logits),tf.zeros_like(attribute_logits))
    prediction = tf.where(specific_logits < 0.5, -tf.ones_like(prediction), prediction)
    prediction = tf.to_int64(prediction, name='predict_label')
        # positive_label = tf.where(valid_males_label == 1.0)

    return prediction


if __name__=='__main__':
    some_var = logits_to_label(attr_logits)
    sess = tf.Session()
    some_var = sess.run(some_var)

    print(some_var)
