import tensorflow as tf

max_align_bytes_op = tf.load_op_library("./max_align_bytes_op.so")
print("max align bytes: ")
sess = tf.Session()
print(sess.run(max_align_bytes_op.max_align_bytes()))
