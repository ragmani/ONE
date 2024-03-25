/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MemoryManager.h"

#include <backend/basic/MemoryPlanner.h>
#include <util/ConfigSource.h>
#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace train
{

TempMemoryManager::TempMemoryManager() : _mem_planner{std::make_unique<basic::FirstFitPlanner>()}
{
  // DO NOTHING
}

void TempMemoryManager::claimPlan(const TempTensorIndex &index, uint32_t size)
{
  _mem_planner->claim(index, size);
}

void TempMemoryManager::releasePlan(const TempTensorIndex &index) { _mem_planner->release(index); }

void TempMemoryManager::allocate(void)
{
  _mem_alloc = std::make_shared<basic::Allocator>(_mem_planner->capacity());
  assert(_mem_alloc->base());
}

uint8_t *TempMemoryManager::getBuffer(const TempTensorIndex &index) const
{
  assert(_mem_planner->memory_plans().find(index) != _mem_planner->memory_plans().end());
  const auto &mem_blk = _mem_planner->memory_plans().at(index);
  return _mem_alloc->base() + mem_blk.offset;
}

} // namespace train
} // namespace backend
} // namespace onert
