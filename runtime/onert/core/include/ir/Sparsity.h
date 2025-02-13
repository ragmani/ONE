/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_SPARSITY_H__
#define __ONERT_IR_SPARSITY_H__

#include <cassert>
#include <cstdint>
#include <vector>

namespace onert::ir
{

/**
 * @brief  Structure for Sparse Tensor
 */
struct Sparsity
{
public:
  Sparsity() = default;
  Sparsity(std::vector<uint16_t> &&w1_segments, std::vector<uint16_t> &&w1_indices,
           std::vector<int32_t> &&block_size)
    : _w1_segments(w1_segments), _w1_indices(w1_indices), _block_size(block_size)
  {
  }

  /**
   * @brief Returns segments array. See compressed sparse row format.
   */
  const uint16_t *w1_segments() const { return _w1_segments.data(); }
  /**
   * @brief Returns indices array. See compressed sparse row format.
   */
  const uint16_t *w1_indices() const { return _w1_indices.data(); }
  /**
   * @brief Returns block size which is used for block sparsity
   */
  const std::vector<int32_t> &block_size() const { return _block_size; }

private:
  std::vector<uint16_t> _w1_segments;
  std::vector<uint16_t> _w1_indices;
  std::vector<int32_t> _block_size;
};

} // namespace onert::ir

#endif // __ONERT_IR_SPARSITY_H__
