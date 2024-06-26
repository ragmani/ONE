/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleDepthToSpace.h"

#include <luci/IR/Nodes/CircleDepthToSpace.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleDepthToSpaceGraphBuilder::validate(const ValidateArgs &args) const
{
  if (!GraphBuilder::validate(args, 1))
    return false;

  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  const auto *options = args.op.builtin_options.AsDepthToSpaceOptions();
  const auto tensors = args.reader.tensors();
  assert(tensors[outputs[0]] != nullptr && tensors[inputs.at(0)] != nullptr);

  if (tensors[outputs[0]]->type() != tensors[inputs.at(0)]->type())
  {
    return false;
  }

  if (options->block_size < 2)
    return false;

  return true;
}

CircleNode *CircleDepthToSpaceGraphBuilder::build_node(const circle::OperatorT &op,
                                                       const std::vector<CircleNode *> &inputs,
                                                       loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleDepthToSpace>();
  node->input(inputs.at(0));

  const auto *options = op.builtin_options.AsDepthToSpaceOptions();
  node->block_size(options->block_size);

  return node;
}

} // namespace luci
