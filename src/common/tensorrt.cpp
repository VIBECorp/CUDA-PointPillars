/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// TensorRT 10.x 포팅: binding index API 제거 -> IO tensor name API 사용
// (JetPack 6.x / TRT 10.3 대응. 기존 index 기반 인터페이스는 IO 텐서 나열 순서로 유지)

#include "tensorrt.hpp"

#include <cuda_runtime.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "check.hpp"

namespace TensorRT {

static class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
} gLogger_;

static std::string format_shape(const nvinfer1::Dims &shape) {
  char buf[200] = {0};
  char *p = buf;
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i + 1 < shape.nbDims)
      p += sprintf(p, "%d x ", (int)shape.d[i]);
    else
      p += sprintf(p, "%d", (int)shape.d[i]);
  }
  return buf;
}

static std::vector<uint8_t> load_file(const std::string &file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

static const char *data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
      return "Float32";
    case nvinfer1::DataType::kHALF:
      return "Float16";
    case nvinfer1::DataType::kINT32:
      return "Int32";
    case nvinfer1::DataType::kINT8:
      return "Int8";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    default:
      return "Unknow";
  }
}

template <typename _T>
static void destroy_pointer(_T *ptr) {
  if (ptr) delete ptr;
}

class __native_engine_context {
 public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size, const char *message_name) {
    destroy();

    if (pdata == nullptr || size == 0) {
      printf("Construct for empty data found.\n");
      return false;
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger_), destroy_pointer<nvinfer1::IRuntime>);
    if (runtime_ == nullptr) {
      printf("Failed to create tensorRT runtime: %s.\n", message_name);
      return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size),
                                                     destroy_pointer<nvinfer1::ICudaEngine>);
    if (engine_ == nullptr) {
      printf("Failed to deserialize engine: %s\n", message_name);
      return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                            destroy_pointer<nvinfer1::IExecutionContext>);
    if (context_ == nullptr) {
      printf("Failed to create execution context: %s\n", message_name);
      return false;
    }
    return context_ != nullptr;
  }

 private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

 public:
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

class EngineImplement : public Engine {
 public:
  std::shared_ptr<__native_engine_context> context_;
  std::unordered_map<std::string, int> binding_name_to_index_;
  std::vector<std::string> binding_names_;

  virtual ~EngineImplement() = default;

  bool construct(const void *data, size_t size, const char *message_name) {
    context_ = std::make_shared<__native_engine_context>();
    if (!context_->construct(data, size, message_name)) {
      return false;
    }

    setup();
    return true;
  }

  bool load(const std::string &file) {
    auto data = load_file(file);
    if (data.empty()) {
      printf("An empty file has been loaded. Please confirm your file path: %s\n", file.c_str());
      return false;
    }
    return this->construct(data.data(), data.size(), file.c_str());
  }

  void setup() {
    auto engine = this->context_->engine_;
    int nbTensors = engine->getNbIOTensors();

    binding_name_to_index_.clear();
    binding_names_.clear();
    for (int i = 0; i < nbTensors; ++i) {
      const char *name = engine->getIOTensorName(i);
      binding_name_to_index_[name] = i;
      binding_names_.push_back(name);
    }
  }

  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s", name.c_str());
    return iter->second;
  }

  virtual bool forward(const std::vector<const void *> &bindings, void *stream, void *input_consum_event) override {
    auto context = this->context_->context_;
    Assertf(bindings.size() == binding_names_.size(), "Invalid bindings count %d, expect %d",
            (int)bindings.size(), (int)binding_names_.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
      if (!context->setTensorAddress(binding_names_[i].c_str(), (void *)bindings[i])) return false;
    }
    if (input_consum_event) {
      context->setInputConsumedEvent(*(cudaEvent_t *)input_consum_event);
    }
    return context->enqueueV3((cudaStream_t)stream);
  }

  virtual std::vector<int> run_dims(const std::string &name) override {
    auto dim = this->context_->context_->getTensorShape(name.c_str());
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<int> run_dims(int ibinding) override { return run_dims(binding_names_[ibinding]); }

  virtual std::vector<int> static_dims(const std::string &name) override {
    auto dim = this->context_->engine_->getTensorShape(name.c_str());
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<int> static_dims(int ibinding) override { return static_dims(binding_names_[ibinding]); }

  virtual int num_bindings() override { return this->context_->engine_->getNbIOTensors(); }

  virtual bool is_input(int ibinding) override {
    return this->context_->engine_->getTensorIOMode(binding_names_[ibinding].c_str()) == nvinfer1::TensorIOMode::kINPUT;
  }

  virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) override {
    nvinfer1::Dims d;
    memcpy(d.d, dims.data(), sizeof(int) * dims.size());
    d.nbDims = dims.size();
    return this->context_->context_->setInputShape(name.c_str(), d);
  }

  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) override {
    return set_run_dims(binding_names_[ibinding], dims);
  }

  virtual int numel(const std::string &name) override {
    auto dim = this->context_->context_->getTensorShape(name.c_str());
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
  }

  virtual int numel(int ibinding) override { return numel(binding_names_[ibinding]); }

  virtual DType dtype(const std::string &name) override {
    return (DType)this->context_->engine_->getTensorDataType(name.c_str());
  }

  virtual DType dtype(int ibinding) override { return dtype(binding_names_[ibinding]); }

  virtual bool has_dynamic_dim() override {
    int numBindings = this->context_->engine_->getNbIOTensors();
    for (int i = 0; i < numBindings; ++i) {
      nvinfer1::Dims dims = this->context_->engine_->getTensorShape(binding_names_[i].c_str());
      for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] == -1) return true;
      }
    }
    return false;
  }

  virtual void print(const char *name) override {
    printf("------------------------------------------------------\n");
    printf("%s 🌱 is %s model\n", name, has_dynamic_dim() ? "Dynamic Shape" : "Static Shape");

    auto engine = this->context_->engine_;
    int nbTensors = engine->getNbIOTensors();
    int num_input = 0;
    int num_output = 0;
    for (int i = 0; i < nbTensors; ++i) {
      if (is_input(i))
        num_input++;
      else
        num_output++;
    }

    printf("Inputs: %d\n", num_input);
    for (int i = 0; i < nbTensors; ++i) {
      if (!is_input(i)) continue;
      auto tname = binding_names_[i].c_str();
      auto dim = engine->getTensorShape(tname);
      auto dtype = engine->getTensorDataType(tname);
      printf("\t%d.%s : {%s} [%s]\n", i, tname, format_shape(dim).c_str(), data_type_string(dtype));
    }

    printf("Outputs: %d\n", num_output);
    for (int i = 0; i < nbTensors; ++i) {
      if (is_input(i)) continue;
      auto tname = binding_names_[i].c_str();
      auto dim = engine->getTensorShape(tname);
      auto dtype = engine->getTensorDataType(tname);
      printf("\t%d.%s : {%s} [%s]\n", i, tname, format_shape(dim).c_str(), data_type_string(dtype));
    }
    printf("------------------------------------------------------\n");
  }
};

std::shared_ptr<Engine> load(const std::string &file) {
  std::shared_ptr<EngineImplement> impl(new EngineImplement());
  if (!impl->load(file)) impl.reset();
  return impl;
}

};  // namespace TensorRT
