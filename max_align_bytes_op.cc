#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("MaxAlignBytes")
    .Output("output: int64")
    .SetIsStateful()
    .Doc(R"doc(Returns EIGEN_MAX_ALIGN_BYTES)doc");

class MaxAlignBytesOp : public OpKernel {
 public:
  explicit MaxAlignBytesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
                                             &output_tensor));
    auto output = output_tensor->flat<int64>();
    
    output(0) = EIGEN_MAX_ALIGN_BYTES;
    
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxAlignBytes").Device(DEVICE_CPU), MaxAlignBytesOp);
