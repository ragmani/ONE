/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Conv2D benchmark with various algorithms (draft version)
 */

#include <nonius/nonius.h++>

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLFunctions.h>

#include <cstdint>
#include <cassert>
#include <stdexcept>

using namespace arm_compute;

//
// Helpers
//
namespace
{

enum Layout
{
  NCHW,
  NHWC
};

struct Initializer
{
  Initializer() { CLScheduler::get().default_init(); }
};

Initializer initializer;

TensorInfo make_info(uint32_t N)
{
  TensorShape shape{N};
  return TensorInfo{shape, 1, DataType::F32};
}

template <enum Layout> TensorInfo make_info(uint32_t N, uint32_t C, uint32_t H, uint32_t W);

template <> TensorInfo make_info<NCHW>(uint32_t N, uint32_t C, uint32_t H, uint32_t W)
{
  TensorShape shape{W, H, C, N};
  TensorInfo info{shape, 1, DataType::F32};
  info.set_data_layout(DataLayout::NCHW);
  return info;
}

template <> TensorInfo make_info<NHWC>(uint32_t N, uint32_t C, uint32_t H, uint32_t W)
{
  TensorShape shape{C, W, H, N};
  TensorInfo info{shape, 1, DataType::F32};
  info.set_data_layout(DataLayout::NHWC);
  return info;
}

inline void check(const Status &status)
{
  if (!status)
  {
    std::cerr << status.error_description() << std::endl;
    throw std::runtime_error{"ERROR"};
  }
}

inline bool is_odd(uint32_t n) { return (n % 2 != 0) ? true : false; }

} // namespace

//
// Benchmark Parameters
//
NONIUS_PARAM(BATCH, 1);

NONIUS_PARAM(IFM_C, 3);
NONIUS_PARAM(IFM_H, 244);
NONIUS_PARAM(IFM_W, 244);

NONIUS_PARAM(OFM_C, 3);
NONIUS_PARAM(OFM_H, 244);
NONIUS_PARAM(OFM_W, 244);

NONIUS_PARAM(KER_H, 3);
NONIUS_PARAM(KER_W, 3);

NONIUS_PARAM(STRIDE_H, 1);
NONIUS_PARAM(STRIDE_W, 1);

NONIUS_PARAM(PADDING, std::string{"SAME"})
NONIUS_PARAM(FUSED_ACT, std::string{"RELU"})

//
// Configuration Helpers
//
namespace
{

struct Configuration
{
  uint32_t ifm_N;
  uint32_t ifm_C;
  uint32_t ifm_H;
  uint32_t ifm_W;

  uint32_t ofm_N;
  uint32_t ofm_C;
  uint32_t ofm_H;
  uint32_t ofm_W;

  uint32_t ker_N;
  uint32_t ker_C;
  uint32_t ker_H;
  uint32_t ker_W;

  uint32_t vertical_stride;
  uint32_t horizontal_stride;

  std::string padding;
  std::string fused_act;

  uint32_t top_padding;
  uint32_t bottom_padding;
  uint32_t left_padding;
  uint32_t right_padding;

  Configuration(nonius::chronometer meter)
  {
    ifm_N = meter.param<BATCH>();
    ifm_C = meter.param<IFM_C>();
    ifm_H = meter.param<IFM_H>();
    ifm_W = meter.param<IFM_W>();

    ofm_N = meter.param<BATCH>();
    ofm_C = meter.param<OFM_C>();
    ofm_H = meter.param<OFM_H>();
    ofm_W = meter.param<OFM_W>();

    ker_N = meter.param<OFM_C>();
    ker_C = meter.param<IFM_C>();
    ker_H = meter.param<KER_H>();
    ker_W = meter.param<KER_W>();

    vertical_stride = meter.param<STRIDE_H>();
    horizontal_stride = meter.param<STRIDE_W>();

    padding = meter.param<PADDING>();
    fused_act = meter.param<FUSED_ACT>();

    assert((ifm_H - ker_H) % vertical_stride == 0);
    assert((ifm_W - ker_H) % horizontal_stride == 0);

    uint32_t const effective_ofm_H = (ifm_H - ker_H) / vertical_stride + 1;
    uint32_t const effective_ofm_W = (ifm_W - ker_H) / horizontal_stride + 1;

    assert(ofm_H >= effective_ofm_H);
    assert(ofm_W >= effective_ofm_W);

    uint32_t const pad_H = ofm_H - effective_ofm_H;
    uint32_t const pad_W = ofm_W - effective_ofm_W;

    top_padding = pad_H / 2;
    bottom_padding = pad_H / 2;
    left_padding = pad_W / 2;
    right_padding = pad_W / 2;

    if (is_odd(pad_H))
      top_padding += 1;
    if (is_odd(pad_W))
      left_padding += 1;
  }

  template <Layout L> TensorInfo src_info() const
  {
    return make_info<L>(ifm_N, ifm_C, ifm_H, ifm_W);
  }
  template <Layout L> TensorInfo dst_info() const
  {
    return make_info<L>(ofm_N, ofm_C, ofm_H, ofm_W);
  }
  template <Layout L> TensorInfo ker_info() const
  {
    return make_info<L>(ker_N, ker_C, ker_H, ker_W);
  }
  TensorInfo bias_info(void) const { return make_info(ker_N); }

  PadStrideInfo pad_stride_info(void) const
  {
    return PadStrideInfo{horizontal_stride,
                         vertical_stride,
                         left_padding,
                         right_padding,
                         top_padding,
                         bottom_padding,
                         DimensionRoundingType::FLOOR};
  }
};

} // namespace

//
// Benchmark Implementations
//
namespace
{

inline nonius::benchmark_registry &local_benchmark_registry()
{
  static nonius::benchmark_registry registry;
  return registry;
}

} // namespace

#define NONIUS_LOCAL_BENCHMARK(name, ...)                                              \
  namespace                                                                            \
  {                                                                                    \
  static ::nonius::benchmark_registrar                                                 \
      NONIUS_DETAIL_UNIQUE_NAME(benchmark_registrar)(local_benchmark_registry(), name, \
                                                     __VA_ARGS__);                     \
  }

NONIUS_LOCAL_BENCHMARK("CLDirectConvolutionLayer_NCHW", [](nonius::chronometer meter) {
  CLDirectConvolutionLayer conv;

  // Configure
  Configuration p{meter};

  CLTensor src_tensor{};
  CLTensor dst_tensor{};
  CLTensor ker_tensor{};
  CLTensor bias_tensor{};

  src_tensor.allocator()->init(p.src_info<NCHW>());
  dst_tensor.allocator()->init(p.dst_info<NCHW>());
  ker_tensor.allocator()->init(p.ker_info<NCHW>());
  bias_tensor.allocator()->init(p.bias_info());

  try
  {
    check(conv.validate(src_tensor.info(), ker_tensor.info(), bias_tensor.info(), dst_tensor.info(),
                        p.pad_stride_info()));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  conv.configure(&src_tensor, &ker_tensor, &bias_tensor, &dst_tensor, p.pad_stride_info());

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  bias_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run!
  meter.measure([&](int) {
    conv.run();
    CLScheduler::get().sync();
  });
})

NONIUS_LOCAL_BENCHMARK("CLDirectConvolutionLayer_NHWC", [](nonius::chronometer meter) {
  CLDirectConvolutionLayer conv;

  // Configure
  Configuration p{meter};

  CLTensor src_tensor{};
  CLTensor dst_tensor{};
  CLTensor ker_tensor{};
  CLTensor bias_tensor{};

  src_tensor.allocator()->init(p.src_info<NHWC>());
  dst_tensor.allocator()->init(p.dst_info<NHWC>());
  ker_tensor.allocator()->init(p.ker_info<NHWC>());
  bias_tensor.allocator()->init(p.bias_info());

  try
  {
    check(conv.validate(src_tensor.info(), ker_tensor.info(), bias_tensor.info(), dst_tensor.info(),
                        p.pad_stride_info()));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  conv.configure(&src_tensor, &ker_tensor, &bias_tensor, &dst_tensor, p.pad_stride_info());

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  bias_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run!
  meter.measure([&](int) {
    conv.run();
    CLScheduler::get().sync();
  });
})

NONIUS_LOCAL_BENCHMARK("CLGEMMConvolutionLayer_NCHW", [](nonius::chronometer meter) {
  CLGEMMConvolutionLayer conv;

  // Configure
  Configuration p{meter};

  CLTensor src_tensor{};
  CLTensor dst_tensor{};
  CLTensor ker_tensor{};
  CLTensor bias_tensor{};

  src_tensor.allocator()->init(p.src_info<NCHW>());
  dst_tensor.allocator()->init(p.dst_info<NCHW>());
  ker_tensor.allocator()->init(p.ker_info<NCHW>());
  bias_tensor.allocator()->init(p.bias_info());

  try
  {
    check(conv.validate(src_tensor.info(), ker_tensor.info(), bias_tensor.info(), dst_tensor.info(),
                        p.pad_stride_info()));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  conv.configure(&src_tensor, &ker_tensor, &bias_tensor, &dst_tensor, p.pad_stride_info());

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  bias_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run
  meter.measure([&](int) {
    conv.run();
    CLScheduler::get().sync();
  });
})

NONIUS_LOCAL_BENCHMARK("CLGEMMConvolutionLayer_NHWC", [](nonius::chronometer meter) {
  CLGEMMConvolutionLayer conv;

  // Configure
  Configuration p{meter};

  CLTensor src_tensor{};
  CLTensor dst_tensor{};
  CLTensor ker_tensor{};
  CLTensor bias_tensor{};

  src_tensor.allocator()->init(p.src_info<NHWC>());
  dst_tensor.allocator()->init(p.dst_info<NHWC>());
  ker_tensor.allocator()->init(p.ker_info<NHWC>());
  bias_tensor.allocator()->init(p.bias_info());

  try
  {
    check(conv.validate(src_tensor.info(), ker_tensor.info(), bias_tensor.info(), dst_tensor.info(),
                        p.pad_stride_info()));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  conv.configure(&src_tensor, &ker_tensor, &bias_tensor, &dst_tensor, p.pad_stride_info());

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  bias_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run
  meter.measure([&](int) {
    conv.run();
    CLScheduler::get().sync();
  });
})

NONIUS_LOCAL_BENCHMARK("CLWinogradConvolutionLayer_NCHW", [](nonius::chronometer meter) {
  CLWinogradConvolutionLayer conv;

  // Configure
  Configuration p{meter};

  CLTensor src_tensor{};
  CLTensor dst_tensor{};
  CLTensor ker_tensor{};
  CLTensor bias_tensor{};

  src_tensor.allocator()->init(p.src_info<NCHW>());
  dst_tensor.allocator()->init(p.dst_info<NCHW>());
  ker_tensor.allocator()->init(p.ker_info<NCHW>());
  bias_tensor.allocator()->init(p.bias_info());

  try
  {
    check(conv.validate(src_tensor.info(), ker_tensor.info(), bias_tensor.info(), dst_tensor.info(),
                        p.pad_stride_info()));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  conv.configure(&src_tensor, &ker_tensor, &bias_tensor, &dst_tensor, p.pad_stride_info());

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  bias_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run
  meter.measure([&](int) {
    conv.run();
    CLScheduler::get().sync();
  });
})

NONIUS_LOCAL_BENCHMARK("CLWinogradConvolutionLayer_NHWC", [](nonius::chronometer meter) {
  CLWinogradConvolutionLayer conv;

  // Configure
  Configuration p{meter};

  CLTensor src_tensor{};
  CLTensor dst_tensor{};
  CLTensor ker_tensor{};
  CLTensor bias_tensor{};

  src_tensor.allocator()->init(p.src_info<NHWC>());
  dst_tensor.allocator()->init(p.dst_info<NHWC>());
  ker_tensor.allocator()->init(p.ker_info<NHWC>());
  bias_tensor.allocator()->init(p.bias_info());

  try
  {
    check(conv.validate(src_tensor.info(), ker_tensor.info(), bias_tensor.info(), dst_tensor.info(),
                        p.pad_stride_info()));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  conv.configure(&src_tensor, &ker_tensor, &bias_tensor, &dst_tensor, p.pad_stride_info());

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  bias_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run
  meter.measure([&](int) {
    conv.run();
    CLScheduler::get().sync();
  });
})

extern "C" nonius::benchmark_registry &benchmark_functions(void)
{
  return local_benchmark_registry();
}
