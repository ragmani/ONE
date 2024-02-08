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

#ifndef __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__

#include <backend/basic/MemoryManager.h>

namespace onert
{
namespace backend
{
namespace train
{

using MemoryManager = backend::basic::MemoryManager;

class TempMemoryManager
{
public:
  TempMemoryManager();
  virtual ~TempMemoryManager() = default;

  void allocate(void);
  uint8_t *getBuffer(const TempTensorIndex &ind) const;
  void deallocate(void) { _mem_alloc->release(); }

  void claimPlan(const TempTensorIndex &ind, uint32_t size);
  void releasePlan(const TempTensorIndex &ind);

private:
  basic::IMemoryPlanner *createMemoryPlanner();

private:
  std::unordered_map<TempTensorIndex, basic::Block> _tensor_mem_map;
  std::shared_ptr<basic::IMemoryPlanner> _mem_planner;
  std::shared_ptr<basic::Allocator> _mem_alloc;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__
