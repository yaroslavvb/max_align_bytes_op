import tensorflow as tf
import sys, os

if __name__=='__main__':
  mydir = os.path.dirname(os.path.abspath(sys.argv[0]))
else:
  mydir = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

so_location = mydir+"/max_align_bytes_op.so"
print("Trying to open", so_location)
max_align_bytes_op = tf.load_op_library(so_location)
sess = tf.Session()
max_align_bytes = sess.run(max_align_bytes_op.max_align_bytes())
print("EIGEN_MAX_ALIGN_BYTES", max_align_bytes)
