/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief
 */

#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "quantize_v2-inl.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::ObjectPtr;
using nnvm::NodeEntry;
using nnvm::Graph;

static inline size_t GetNumOutputs(ObjectPtr node) {
  // Get NumOutputs, check if current node has NumVisibleOutputs function, if yes, return
  // num_visible_outputs
  size_t num_outputs = node->num_outputs();
  static const auto& num_visible_outputs_attr =
      nnvm::Op::GetAttr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs");
  auto num_visible_output_func = num_visible_outputs_attr.get(node->op(), nullptr);
  if (num_visible_output_func != nullptr) {
    num_outputs = num_visible_output_func(node->attrs);
  }
  return num_outputs;
}

static ObjectPtr CreateNode(std::string op_name, std::string node_name) {
  ObjectPtr node = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed =
      nnvm::Symbol::CreateVariable(node->attrs.name).outputs[0].node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

/*!
 * \brief Insert a node named with node_name holding the op of op_name
 * before the node current and after the node previous.
 */
static ObjectPtr InsertNode(std::string op_name,
    std::string node_name, ObjectPtr current, NodeEntry previous) {
  ObjectPtr node = CreateNode(op_name, node_name);
  node->inputs.emplace_back(previous);
  current->inputs.emplace_back(node);
  return node;
}

static std::vector<NodeEntry> OfflineParams(std::vector<NodeEntry>&& outputs,
                                     const std::unordered_set<std::string>& offline_params) {
  std::string node_suffixs[3] = {"", "_min", "_max"};
  std::unordered_map<Node*, ObjectPtr> mirror_map;
  nnvm::NodeEntryMap<ObjectPtr> entry_var;
  auto need_offline = [&](ObjectPtr n) {
    return (n->op() == Op::Get("_contrib_quantize_v2")) &&
           n->inputs[0].node->is_variable() &&
           offline_params.count(n->inputs[0].node->attrs.name);
  };
  DFSVisit(outputs, [&](const ObjectPtr& node) {
    for (NodeEntry& e : node->inputs) {
      if (need_offline(e.node)) {
        std::string node_name = e.node->attrs.name;
        if (!entry_var.count(e)) {
          entry_var[e] = CreateNode("nullptr", node_name + node_suffixs[e.index]);
        }
        e.node = entry_var[e];
        e.index = 0;
        e.version = 0;
      }
    }
  });
  return std::move(outputs);
}

// To check if a node is registered with a computation function on a target device.
static bool isRegistered(ObjectPtr node, const int& dev_type) {
  const auto& op = node->op();
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), 0);
  FCompute fcompute = common::GetFCompute<FCompute>(op, "FCompute", ctx);
  FComputeEx fcomp_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", ctx);
  FStatefulCompute fcomputestateful =
      common::GetFCompute<FStatefulCompute>(op, "FStatefulCompute", ctx);
  FStatefulComputeEx fcomputestateful_ex =
      common::GetFCompute<FStatefulComputeEx>(op, "FStatefulComputeEx", ctx);
  return (fcompute != nullptr || fcomp_ex != nullptr ||
          fcomputestateful != nullptr || fcomputestateful_ex != nullptr);
}

inline QuantizeType NeedQuantize(ObjectPtr node,
                                 const std::unordered_set<std::string>& excluded_nodes,
                                 const std::unordered_set<std::string>& excluded_ops,
                                 const int& dev_type,
                                 std::unordered_map<ObjectPtr, ObjectPtr>* quantized_node_map,
                                 const std::string quantize_granularity) {
  std::unordered_map<ObjectPtr, ObjectPtr> quantized_node;
  static auto& quantizable_map = Op::GetAttr<mxnet::FQuantizable>("FQuantizable");
  static auto& quantized_op_map = Op::GetAttr<mxnet::FQuantizedOp>("FQuantizedOp");
  static auto& fexec_type = nnvm::Op::GetAttr<FExecType>("FExecType");
  const auto& op = node->op();
  bool need = false;
  if (op && quantized_op_map.count(op)) {
    need = true;
    // If the quantized node is not registered with a computation function, the node
    // will be excluded automatically.
    auto q_ptr = quantized_op_map[node->op()];
    auto qnode = q_ptr(node->attrs);
    if (!isRegistered(qnode, dev_type)) {
      LOG(INFO) << "Neither FCompute nor FComputeEx registered, " << node->op()->name
                << " is excluded automatically.";
      need = false;
    } else {
      if (excluded_nodes.count(node->attrs.name) ||
          excluded_ops.count(node->op()->name)) {
        need = false;
      } else if (!node->attrs.subgraphs.empty()) {
        ExecType exec_type = fexec_type.count(op) ? fexec_type[op](node->attrs) : ExecType::kSync;
        if (exec_type != ExecType::kSubgraphExec) {
          // This is a fused subgraph node, try to match inner node.
          CHECK_EQ(node->attrs.subgraphs.size(), 1);
          auto subgraph_sym = node->attrs.subgraphs[0];
          DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr& n) {
            if (n->is_variable()) return;
            if (excluded_nodes.count(n->attrs.name)) {
              need = false;
            }
          });
        }
      }
    }
    if (need) {
      auto quantized_node = quantized_op_map[op](node->attrs);
      if (!quantized_node->op()) need = false;
      if (need) {
        if ((quantize_granularity == "channel-wise") &&
            (node->op() == Op::Get("_sg_mkldnn_fully_connected"))) {
          quantized_node->attrs.dict["channel_wise_quantize"] = "True";
        }
        quantized_node_map->insert(std::make_pair(node, quantized_node));
      }
      if (quantizable_map.count(op)) {
        return quantizable_map[op](node->attrs);
      } else {
        return QuantizeType::kSupport;
      }
    }
  }
  CHECK(!need);
  return QuantizeType::kNone;
}

enum quantize_bit {
  kFromInput = 1,
  kFromOutput = 2,
};



Graph EmbedQuantizeGraph(Graph &&src) {
  static const auto& flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  static const auto& need_requantize_map = Op::GetAttr<mxnet::FNeedRequantize>("FNeedRequantize");
  static const auto& avoid_quantize_input_map =
      Op::GetAttr<mxnet::FAvoidQuantizeInput>("FAvoidQuantizeInput");
  const auto offline_params = src.GetAttr<std::unordered_set<std::string>>("offline_params");
  const auto quantized_dtype = src.GetAttr<std::string>("quantized_dtype");
  const auto quantize_granularity = src.GetAttr<std::string>("quantize_granularity");
  const auto dev_type = src.GetAttr<int>("target_ctx");

  const auto excluded_nodes = src.GetAttr<std::unordered_set<std::string>>("excluded_nodes");
  const auto excluded_ops = src.GetAttr<std::unordered_set<std::string>>("excluded_ops");


  if (dev_type == Context::kGPU && quantize_granularity == "channel-wise") {
    LOG(FATAL) << "`channel-wise` quantization option is not supported yet by GPU,"
               << " please set quantize_granularity to `tensor-wise` when quantizing model.";
  }

  std::unordered_map<Node*, ObjectPtr> mirror_map;
  std::unordered_map<ObjectPtr, ObjectPtr> reverse_mirror_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_entry_map;
  static int verbose = dmlc::GetEnv("MXNET_QUANTIZATION_VERBOSE", 0);

  DFSVisit(src.outputs, [&](const ObjectPtr& node)
  {
      ObjectPtr new_node = Node::Create();

      *new_node = *node;
      new_node->inputs.clear();

      for (const auto &e : node->inputs)
      {
        ObjectPtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
        if (mirror_entry_map.count(e))
        {
          new_node->inputs.emplace_back(
              mirror_entry_map[e].node->inputs[0].node, e.index, e.version);
        }
        else
        {
          new_node->inputs.emplace_back(mirror_node, e.index, e.version);
        }
      }

      if(node->is_variable()
          || excluded_nodes.count(node->attrs.name)
          || excluded_ops.count(node->op()->name))
      {
        if(verbose)
          LOG(INFO) << "skip node: " << node->attrs.name << " input size:" << node->inputs.size();
      }
      else
      {
        if(verbose)
          LOG(INFO) << "add qualiazaton dequantize  layer" << node->attrs.name << " input size:" << node->inputs.size();
        ObjectPtr quantize_node = Node::Create();
        quantize_node->attrs.op = Op::Get("_contrib_quantize_v2");
        quantize_node->attrs.name = "quantize_" + node->attrs.name;
        quantize_node->attrs.dict["out_type"] = quantized_dtype;
        if (quantize_node->op()->attr_parser != nullptr) {
          quantize_node->op()->attr_parser(&(quantize_node->attrs));
        }
        quantize_node->inputs.emplace_back(new_node, static_cast<uint32_t>(0), 0);

        new_node = quantize_node;
        reverse_mirror_map[quantize_node] = node;

        ObjectPtr requantize_node = Node::Create();
        requantize_node->attrs.op = Op::Get("_contrib_dequantize");
        requantize_node->attrs.name = "dequantize_" + node->attrs.name;
        if (requantize_node->op()->attr_parser != nullptr) {
          requantize_node->op()->attr_parser(&(requantize_node->attrs));
        }
        for (size_t i = 0; i < 3; ++i) {
          requantize_node->inputs.emplace_back(new_node, static_cast<uint32_t>(i), 0);
        }
        new_node = requantize_node;
      }

      mirror_map[node.get()] = new_node;
      reverse_mirror_map[new_node] = node;
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
      outputs.emplace_back(mirror_map.at(e.node.get()), e.index, e.version);
  }

  if (!offline_params.empty()) outputs = OfflineParams(std::move(outputs), offline_params);

  Graph ret;
  ret.outputs = std::move(outputs);

  static const auto& need_calib_input_map =
      Op::GetAttr<mxnet::FNeedCalibrateInput>("FNeedCalibrateInput");
  static const auto& need_calib_output_map =
      Op::GetAttr<mxnet::FNeedCalibrateOutput>("FNeedCalibrateOutput");
  std::vector<std::string> calib_nodes;
  DFSVisit(ret.outputs, [&](const ObjectPtr& node) {
      if (need_calib_input_map.count(node->op())) {
        const auto calib_idx = need_calib_input_map[node->op()](node->attrs);
        for (const auto &idx : calib_idx) {
          if (reverse_mirror_map.count(node)) {
            calib_nodes.push_back(common::GetOutputName(
                {reverse_mirror_map[node], node->inputs[idx].index, node->inputs[idx].version}));
          } else {
            const auto& e = node->inputs[idx];
            if (e.node->is_variable()) {
              calib_nodes.push_back(e.node->attrs.name);
            } else {
              if (reverse_mirror_map.count(e.node)) {
                const auto& fp32_in_node = reverse_mirror_map.at(e.node);
                calib_nodes.push_back(common::GetOutputName({fp32_in_node, e.index, e.version}));
              } else {
                LOG(FATAL) << "Can't find calibration node for " << node->attrs.name;
              }
            }
          }
        }
      } else if (need_calib_output_map.count(node->op())) {
        const auto calib_idx = need_calib_output_map[node->op()](node->attrs);
        for (const auto& idx : calib_idx) {
          if (reverse_mirror_map.count(node)) {
            calib_nodes.push_back(
                common::GetOutputName({reverse_mirror_map[node], static_cast<uint32_t>(idx), 0}));
          } else {
            calib_nodes.push_back(common::GetOutputName({node, static_cast<uint32_t>(idx), 0}));
          }
        }
      }
  });

  ret.attrs["calib_nodes"] = std::make_shared<dmlc::any>(std::move(calib_nodes));

  return ret;
}


NNVM_REGISTER_PASS(EmbedQuantizeGraph)
.describe("")
.set_body(EmbedQuantizeGraph)
.provide_graph_attr("calib_nodes")
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
