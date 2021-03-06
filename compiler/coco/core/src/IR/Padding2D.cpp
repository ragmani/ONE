/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "coco/IR/Padding2D.h"

namespace coco
{

Padding2D &Padding2D::top(uint32_t value)
{
  _top = value;
  return (*this);
}

Padding2D &Padding2D::bottom(uint32_t value)
{
  _bottom = value;
  return (*this);
}

Padding2D &Padding2D::left(uint32_t value)
{
  _left = value;
  return (*this);
}

Padding2D &Padding2D::right(uint32_t value)
{
  _right = value;
  return (*this);
}

} // namespace coco
