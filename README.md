TensorFlow op to return EIGEN_MAX_ALIGN_BYTES

To compile on Mac:

    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    g++ -march=native -std=c++11 -undefined dynamic_lookup -shared max_align_bytes_op.cc -o max_align_bytes_op.so -fPIC -I $TF_INC -O2

To compile on Linux:

```
Follow instructions in https://github.com/tensorflow/tensorflow/issues/12482#issuecomment-328829250
then
export op=max_align_bytes_op
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -march=native -std=c++11 -shared $op.cc -o $op.so -fPIC -I $TF_INC -L$TF_LIB -ltensorflow_framework -O2
python max_align_bytes_op_run.py
```

Then in the same directory:
```
python
import tensorflow as tf
max_align_bytes_op = tf.load_op_library("./max_align_bytes_op.so")
print("max align bytes: ")
sess = tf.Session()
print(sess.run(max_align_bytes_op.max_align_bytes())
```
