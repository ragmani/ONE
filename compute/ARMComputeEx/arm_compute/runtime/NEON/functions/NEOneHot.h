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

/*
 * Copyright (c) 2019 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_NEONEHOT_H__
#define __ARM_COMPUTE_NEONEHOT_H__
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"
namespace arm_compute
{
// Forward declarations
class ITensor;
/** Basic function to run @ref NEOneHotKernel */
class NEOneHot : public INESimpleFunctionNoBorder
{
public:
  /** Initialise the kernel's inputs and outputs
   *
   * @param[in]  indices   Indices tensor. Supported tensor rank: up to 3. Must be one of the
   * following types: U32/S32
   * @param[in]  depth     The tensor for depth of the one hot dimension. Supported tensor rank: up
   * to 3. Must be one of the following types: U32/S32
   * @param[in]  on_value  On value tensor. Supported tensor rank: only 1. Data type supported:
   * U8/S8/U16/S16/F16/U32/S32/F32
   * @param[in]  off_value Off value tensor. Supported tensor rank: only 1. Data type supported:
   * Same as @p on_value
   * @param[out] output    Destination tensor. Data type supported: Same as @p on_value
   * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
   * The value must be in range [-indices.rank , indices.rank)
   */
  void configure(const ITensor *indices, const ITensor *depth, const ITensor *on_value,
                 const ITensor *off_value, ITensor *output, int axis = -1);
  /** Static function to check if given info will lead to a valid configuration of @ref
 * NEOneHotKernel
   *
 * @param[in]  indices   Indices tensor info. Supported tensor rank: up to 3. Must be one of the
 * following types: U32/S32
 * @param[in]  depth     The tensor info for depth of the one hot dimension. Supported tensor rank:
 * up to 3. Must be one of the following types: U32/S32
 * @param[in]  on_value  On value tensor info. Supported tensor rank: only 1. Data type supported:
 * U8/S8/U16/S16/F16/U32/S32/F32
 * @param[in]  off_value Off value tensor info. Supported tensor rank: only 1. Data type supported:
 * Same as @p on_value
 * @param[out] output    Destination tensor info. Data type supported: Same as @p on_value
 * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
 * The value must be in range [-indices.rank , indices.rank)
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *indices, const ITensorInfo *depth,
                         const ITensorInfo *on_value, const ITensorInfo *off_value,
                         const ITensorInfo *output, int axis = -1);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEONEHOT_H__ */
