#ifndef BMNN_H
#define BMNN_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tpuv7_modelrt.h"
#include "tpuv7_rt.h"

namespace {
#define ASSERT(cond) \
  do {               \
    if (!cond) {     \
      exit(1);       \
    }                \
  } while (false)
}  // namespace

/*
 * Any class that inherits this class cannot be assigned.
 */
class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable& rhs) = delete;
};

using tensorSizeType = unsigned long long;

tensorSizeType getSize(const tpuRtTensor_t tensor) {
  // TODO: assert tensor!=nullptr
  tensorSizeType ret = 1;
  for (int i = 0; i < tensor.shape.num_dims; ++i) {
    ret *= tensor.shape.dims[i];
  }
  switch (tensor.dtype) {
    case TPU_FLOAT32:
    case TPU_INT32:
    case TPU_UINT32:
      ret *= 4;
      break;
    case TPU_FLOAT16:
    case TPU_UINT16:
    case TPU_INT16:
    case TPU_BFLOAT16:
      ret *= 2;
      break;
    case TPU_INT8:
    case TPU_UINT8:
      break;
    case TPU_INT4:
    case TPU_UINT4:
      ret /= 2;
      break;
  }
  return ret;
}

class BMNNTensor {
 public:
  BMNNTensor(const char* name, float scale, tpuRtTensor_t* tensor,
             tpuRtStream_t* stream)
      : m_name(name),
        m_host_data(nullptr),
        m_scale(scale),
        m_tensor(tensor),
        stream(stream) {}

  virtual ~BMNNTensor() {
    if (m_host_data) {
      delete m_host_data;
    }
  }

  // Return an array pointer to system memory of tensor.
  void* get_host_data() {
    if (m_host_data) return m_host_data;
    tpuRtStatus_t ret;
    auto size = getSize(*m_tensor);
    m_host_data = new char[size];  // bytes
    tpuRtMemcpyD2SAsync(m_host_data, m_tensor->data, size, *stream);
    tpuRtStreamSynchronize(*stream);
    return m_host_data;
  }

  const tpuRtShape_t* get_shape() { return &m_tensor->shape; }

  tpuRtDataType_t get_dtype() { return m_tensor->dtype; }

  float get_scale() { return m_scale; }

 private:
  std::string m_name;
  void* m_host_data;
  float m_scale;
  tpuRtTensor_t* m_tensor;
  tpuRtStream_t* stream;
};

class BMNNNetwork : public NoCopyable {
 private:
  tpuRtTensor_t* m_inputTensors;
  tpuRtTensor_t* m_outputTensors;
  int m_max_batch;
  tpuRtNet_t* net;
  // tpuRtNetInfo_t* m_netinfo;
  tpuRtNetInfo_t m_netinfo;
  tpuRtStream_t stream;

 public:
  BMNNNetwork(tpuRtNet_t* netPtr) : net(netPtr) {
    m_netinfo = tpuRtGetNetInfo(netPtr);
    // m_netinfo = &info;
    tpuRtStreamCreate(&stream);
    m_inputTensors = new tpuRtTensor_t[m_netinfo.input.num];
    m_outputTensors = new tpuRtTensor_t[m_netinfo.output.num];
    m_max_batch = -1;
    for (int i = 0; i < m_netinfo.stage_num; i++) {
      int b = m_netinfo.stages[i].input_shapes[0].dims[0];
      m_max_batch = m_max_batch > b ? m_max_batch : b;
    }
    for (int i = 0; i < m_netinfo.input.num; ++i) {
      m_inputTensors[i].dtype = m_netinfo.input.dtypes[i];
      m_inputTensors[i].shape = m_netinfo.stages[0].input_shapes[i];
      //   m_inputTensors[i].device_mem = bm_mem_null();
    }
    for (int i = 0; i < m_netinfo.output.num; ++i) {
      m_outputTensors[i].dtype = m_netinfo.output.dtypes[i];
      m_outputTensors[i].shape = m_netinfo.stages[0].output_shapes[i];
      tensorSizeType max_size = 0;
      for (int s = 0; s < m_netinfo.stage_num; s++) {
        tensorSizeType out_size = getSize(m_outputTensors[i]);
        if (max_size < out_size) {
          max_size = out_size;
        }
      }
      auto ret = tpuRtMalloc(&m_outputTensors[i].data, max_size, 0);
      ASSERT(ret == 0);
      // assert()
    }
    showInfo();
  }

  ~BMNNNetwork() {
    tpuRtStreamDestroy(stream);
    delete[] m_inputTensors;
    for (int i = 0; i < m_netinfo.output.num; ++i) {
      tpuRtFree(&m_outputTensors[i].data, 0);
    }
    delete[] m_outputTensors;
  }

  int maxBatch() const { return m_max_batch; }

  std::shared_ptr<BMNNTensor> inputTensor(int index, int stage_idx = -1) {
    if (stage_idx >= 0) {
      for (int i = 0; i < m_netinfo.input.num; ++i) {
        m_inputTensors[i].shape = m_netinfo.stages[stage_idx].input_shapes[i];
      }
    }
    return std::make_shared<BMNNTensor>(m_netinfo.input.names[index],
                                        m_netinfo.input.scales[index],
                                        &m_inputTensors[index], &stream);
  }

  int outputTensorNum() { return m_netinfo.output.num; }

  std::shared_ptr<BMNNTensor> outputTensor(int index, int stage_idx = -1) {
    if (stage_idx >= 0) {
      for (int i = 0; i < m_netinfo.output.num; ++i) {
        m_outputTensors[i].shape = m_netinfo.stages[stage_idx].output_shapes[i];
      }
    }
    return std::make_shared<BMNNTensor>(m_netinfo.output.names[index],
                                        m_netinfo.output.scales[index],
                                        &m_outputTensors[index], &stream);
  }

  tpuRtStatus_t forward() {
    tpuRtStatus_t ret;
    ret = tpuRtLaunchNet(net, m_inputTensors, m_outputTensors, m_netinfo.name,
                         stream);
    return ret;
  }

  tpuRtStatus_t forward(
      std::vector<std::shared_ptr<tpuRtTensor_t>>& inputTensors,
      std::vector<std::shared_ptr<tpuRtTensor_t>>& outputTensors) {
    tpuRtTensor_t tempInputTensors[m_netinfo.input.num];
    for (int i = 0; i < m_netinfo.input.num; ++i)
      tempInputTensors[i] = *inputTensors[i];
    tpuRtTensor_t tempOutputTensors[m_netinfo.output.num];
    for (int i = 0; i < m_netinfo.output.num; ++i)
      tempOutputTensors[i] = *outputTensors[i];

    tpuRtStatus_t ret;
    ret = tpuRtLaunchNet(const_cast<tpuRtNet_t*>(net), tempInputTensors,
                         tempOutputTensors, m_netinfo.name, stream);
    return ret;
  }

  tpuRtStatus_t forwardAsync() {
    tpuRtStatus_t ret;
    ret = tpuRtLaunchNetAsync(const_cast<tpuRtNet_t*>(net), m_inputTensors,
                              m_outputTensors, m_netinfo.name, stream);
    return ret;
  }

  tpuRtStatus_t forwardAsync(
      std::vector<std::shared_ptr<tpuRtTensor_t>>& inputTensors,
      std::vector<std::shared_ptr<tpuRtTensor_t>>& outputTensors) {
    tpuRtTensor_t tempInputTensors[m_netinfo.input.num];
    for (int i = 0; i < m_netinfo.input.num; ++i)
      tempInputTensors[i] = *inputTensors[i];
    tpuRtTensor_t tempOutputTensors[m_netinfo.output.num];
    for (int i = 0; i < m_netinfo.output.num; ++i)
      tempOutputTensors[i] = *outputTensors[i];

    tpuRtStatus_t ret;
    ret = tpuRtLaunchNetAsync(const_cast<tpuRtNet_t*>(net), tempInputTensors,
                              tempOutputTensors, m_netinfo.name, stream);
    return ret;
  }

  static std::string shape_to_str(const tpuRtShape_t& shape) {
    std::string str = "[ ";
    for (int i = 0; i < shape.num_dims; i++) {
      str += std::to_string(shape.dims[i]) + " ";
    }
    str += "]";
    return str;
  }

  void showInfo() {
    const char* dtypeMap[] = {"FLOAT32",  "FLOAT16", "INT8",  "UINT8",
                              "INT16",    "UINT16",  "INT32", "UINT32",
                              "BFLOAT16", "INT4",    "UINT4"};
    printf("\n########################\n");
    printf("NetName: %s\n", m_netinfo.name);
    for (int s = 0; s < m_netinfo.stage_num; s++) {
      printf("---- stage %d ----\n", s);
      for (int i = 0; i < m_netinfo.input.num; i++) {
        auto shapeStr = shape_to_str(m_netinfo.stages[s].input_shapes[i]);
        printf("  Input %d) '%s' shape=%s dtype=%s scale=%g\n", i,
               m_netinfo.input.names[i], shapeStr.c_str(),
               dtypeMap[m_netinfo.input.dtypes[i]], m_netinfo.input.scales[i]);
      }
      for (int i = 0; i < m_netinfo.output.num; i++) {
        auto shapeStr = shape_to_str(m_netinfo.stages[s].output_shapes[i]);
        printf("  Output %d) '%s' shape=%s dtype=%s scale=%g\n", i,
               m_netinfo.output.names[i], shapeStr.c_str(),
               dtypeMap[m_netinfo.output.dtypes[i]],
               m_netinfo.output.scales[i]);
      }
    }
    printf("########################\n\n");
  }
};

/*
 * Help user managing handles and networks of a bmodel, using class instances
 * above.
 */
class BMNNContext : public NoCopyable {
  tpuRtNet_t net;
  tpuRtNetContext_t context;
  std::vector<std::string> m_network_names;

 public:
  BMNNContext(const char* bmodel_file) {
    auto ret = tpuRtCreateNetContext(&context);
    ret = tpuRtLoadNet(bmodel_file, context, &net);
    if (ret != tpuRtSuccess) {
      std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
    }
  }

  ~BMNNContext() {
    tpuRtUnloadNet(&net);
    // tpuRtDestroyNetContext(context);
  }

  std::string network_name(int index) {
    if (index >= (int)m_network_names.size()) {
      return "Invalid index";
    }

    return m_network_names[index];
  }

  std::shared_ptr<BMNNNetwork> network() {
    return std::make_shared<BMNNNetwork>(&net);
  }

  //   std::shared_ptr<BMNNNetwork> network() {
  //     return std::make_shared<BMNNNetwork>();
  //   }
};

#endif