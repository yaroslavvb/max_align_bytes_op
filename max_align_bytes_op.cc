#include <limits.h>
#include <atomic>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

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
    
    AllocatorAttributes alloc_attrs;
    auto device = static_cast<tensorflow::Device*>(ctx->device());
    Allocator* allocator = device->GetAllocator(alloc_attrs);
    // this doesn't do anything until cpu_allocator_collect_full_stats
    // is marked as extern
    EnableCPUAllocatorStats(true);
    AllocatorStats stats;
    allocator->GetStats(&stats);
    output(0) = EIGEN_MAX_ALIGN_BYTES;
    
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxAlignBytes").Device(DEVICE_CPU), MaxAlignBytesOp);
