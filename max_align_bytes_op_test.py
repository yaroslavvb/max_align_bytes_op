import tensorflow as tf

max_align_bytes_op = tf.load_op_library("./max_align_bytes_op.so")
sess = tf.Session()
max_align_bytes = sess.run(max_align_bytes_op.max_align_bytes())
print("EIGEN_MAX_ALIGN_BYTES", max_align_bytes)
