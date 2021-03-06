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

#ifndef __ONERT_EXEC_OBSREVERS_H__
#define __ONERT_EXEC_OBSREVERS_H__

#include "exec/IFunction.h"
#include "ir/OpSequence.h"
#include "ExecTime.h"
#include "util/ITimer.h"
#include "exec/IExecutor.h"
#include "util/EventCollector.h"
#include "util/EventRecorder.h"

namespace onert
{
namespace exec
{
class IExecutionObserver
{
public:
  /// @brief Invoked just before model (not individual operation) execution begins
  virtual void handleSubgraphBegin(IExecutor *) { return; }

  virtual void handleJobBegin(IExecutor *, const ir::OpSequence *, const backend::Backend *) = 0;
  virtual void handleJobEnd(IExecutor *, const ir::OpSequence *, const backend::Backend *) = 0;

  /// @brief Invoked just after model (not individual operation) execution ends
  virtual void handleSubgraphEnd(IExecutor *) { return; }

  virtual ~IExecutionObserver() = default;
};

class ProfileObserver : public IExecutionObserver
{
public:
  explicit ProfileObserver(std::shared_ptr<ExecTime> et, const ir::Graph &graph)
      : _et(std::move(et)), _graph(graph)
  {
  }
  void handleJobBegin(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;
  void handleJobEnd(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;

  void handleSubgraphEnd(IExecutor *) override { _et->uploadOperationsExecTime(); }

private:
  std::unique_ptr<util::ITimer> _timer;
  std::shared_ptr<ExecTime> _et;
  const ir::Graph &_graph;
};

class ChromeTracingObserver : public IExecutionObserver
{
public:
  ChromeTracingObserver(const std::string &filepath, const ir::Graph &graph);
  ~ChromeTracingObserver();
  void handleSubgraphBegin(IExecutor *) override;
  void handleJobBegin(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;
  void handleJobEnd(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;
  void handleSubgraphEnd(IExecutor *) override;

private:
  static std::string opSequenceTag(const ir::OpSequence *op_seq, const ir::Operations &operations);

private:
  const std::string &_base_filepath;
  EventRecorder _recorder;
  EventCollector _collector;
  const ir::Graph &_graph;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_OBSREVERS_H__
