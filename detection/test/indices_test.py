
import numpy as np
import tensorflow as tf

males =tf.constant(np.array([1, 12, 14, 6, 1, 9, 3, 1, 4, 6]))
male_logits = tf.constant(np.array([(0.4, 0.5), (0.72, 0.58), (0.16, 0.84), (0.77, 0.83), (0.51, 0.49),(0.4, 0.4), (0.72, 0.58), (0.84, 0.16), (0.77, 0.83), (0.49, 0.51)]))



indices = tf.stack([tf.range(tf.size(males)), tf.to_int32(males) - 1], axis=1)


sess = tf.Session()
loss = sess.run(indices)

print(loss)
