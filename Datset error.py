import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset

#
# one_tensor = tf.ones([1,28,28,1])
#
# reshaped_tensor = tf.reshape(one_tensor,[-1,28])
#
# print(reshaped_tensor)
#
# with tf.Session() as sess:
#     print(one_tensor.eval())
#

#
placeholder = tf.placeholder(dtype=tf.float32, shape=[2,30])

dataset = Dataset.from_tensor_slices((placeholder))

print(dataset.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset.output_shapes)  # ==> "((), (100,))

iterator = dataset.make_initializable_iterator()

next_example = iterator.get_next()

final = tf.reshape(next_example,[-1,5])
print(final)

print(final)

a = np.ones([2,30])

with tf.Session() as sess:
    sess.run(iterator.initializer,feed_dict={placeholder:a})
    print(final.eval())