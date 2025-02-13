/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_CPU_OPS_GATHERLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_GATHERLAYER_H__

#include "../ExternalContext.h"

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert::backend::cpu::ops
{

class GatherLayer : public ::onert::exec::IFunction
{
public:
  GatherLayer() : _input{nullptr}, _indices{nullptr}, _output{nullptr}, _axis{-1}, _ctx{nullptr}
  {
    // DO NOTHING
  }

public:
  void configure(const IPortableTensor *input, const IPortableTensor *indices,
                 IPortableTensor *output, int32_t axis, ExternalContext *ctx);

  void run() override;

private:
  template <typename OpType> void runByInputType();
  void runByGGMLQuantInputType();

private:
  const IPortableTensor *_input;
  const IPortableTensor *_indices;
  IPortableTensor *_output;

  int32_t _axis;
  ExternalContext *_ctx;
};

} // namespace onert::backend::cpu::ops

#endif // __ONERT_BACKEND_CPU_OPS_GATHERLAYER_H__
