TensorFlow op to return EIGEN_MAX_ALIGN_BYTES

To compile on Mac:

    g++ -std=c++11 -undefined dynamic_lookup -shared max_align_bytes_op.cc -o max_align_bytes_op.so -fPIC -I $TF_INC -O2

Then in the same directory:

    max_align_bytes_op = tf.load_op_library("./max_align_bytes_op.so")
    print("max align bytes: ")
    sess = tf.Session()
    print(sess.run(max_align_bytes_op.max_align_bytes())
