/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_REGISTRY__
#define __ONERT_BACKEND_TRAIN_TENSOR_REGISTRY__

#include <backend/train/ITensorRegistry.h>

#include "TempTensorIndex.h"
#include "Tensor.h"

namespace onert
{
namespace backend
{
namespace train
{

class TensorRegistry
  : public PortableTensorRegistryTemplate<Tensor, TrainableTensor, BackPropTensor, GradientTensor>
{
public:
  TempTensor *getTempTensor(const TempTensorIndex &index)
  {
    auto tensor = _temp.find(index);
    if (tensor != _temp.end())
      return tensor->second.get();
    return nullptr;
  }

  void setTempTensor(const TempTensorIndex &index, std::unique_ptr<TempTensor> tensor)
  {
    assert(tensor != nullptr);
    auto itr = _temp.find(index);
    if (itr != _temp.end())
      throw std::runtime_error{
        "Tried to set a temp tensor but another temp tensor already exists."};

    _temp[index] = std::move(tensor);
  }

  const std::unordered_map<TempTensorIndex, std::unique_ptr<TempTensor>> &temp_tensors()
  {
    return _temp;
  }

private:
  std::unordered_map<TempTensorIndex, std::unique_ptr<TempTensor>> _temp;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_TENSOR_REGISTRY__
