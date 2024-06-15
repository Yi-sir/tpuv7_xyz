#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

#include "tpu_utils.h"

using namespace std;

float getDiff(char** outBuffer, char** groundTruth, std::vector<int>& dims) {
  float ret = 0.0;
  float** outBufferFloat = reinterpret_cast<float**>(outBuffer);
  float** groundTruthFloat = reinterpret_cast<float**>(groundTruth);
  std::vector<std::vector<float>> absDiff(dims.size());

  for (int i = 0; i < dims.size(); ++i) {
    int len = dims[i] / sizeof(float);
    for (int j = 0; j < len; ++j) {
      absDiff[i].push_back(
          std::abs(outBufferFloat[i][j] - groundTruthFloat[i][j]));
    }
    ret += std::accumulate(absDiff[i].begin(), absDiff[i].end(), 0.0);
  }
  return ret;
}

std::vector<int> dims{1632000 * 4, 408000 * 4, 102000 * 4};

int main() {
  long inSize, outSize;
  char* inBuffer;
  char** outBuffer = new char*[3];
  char** fileOutBuffer = new char*[3];
  std::ifstream file1("../input_ref_data.dat.bmrt",
                      std::ios::binary | std::ios::ate);
  if (file1.is_open()) {
    std::streampos fileSize = file1.tellg();
    file1.seekg(0, std::ios::beg);
    inSize = fileSize;
    inBuffer = new char[inSize];
    file1.read(inBuffer, inSize);
    file1.close();
  } else {
    std::cerr << "无法打开文件" << std::endl;
  }
  outBuffer[0] = new char[dims[0]];
  outBuffer[1] = new char[dims[1]];
  outBuffer[2] = new char[dims[2]];

  std::ifstream file2("../output_ref_data.dat.bmrt",
                      std::ios::binary | std::ios::ate);
  if (file2.is_open()) {
    std::streampos fileSize = file2.tellg();
    outSize = fileSize;

    fileOutBuffer[0] = new char[dims[0]];
    fileOutBuffer[1] = new char[dims[1]];
    fileOutBuffer[2] = new char[dims[2]];

    file2.seekg(0, std::ios::beg);
    file2.read(fileOutBuffer[0], dims[0]);
    // file2.seekg(dims[0], std::ios::beg);
    file2.read(fileOutBuffer[1], dims[1]);
    // file2.seekg(dims[0] + dims[1], std::ios::beg);
    file2.read(fileOutBuffer[2], dims[2]);

    file2.close();
  } else {
    std::cerr << "无法打开文件" << std::endl;
  }

  std::cout << inSize << std::endl << outSize << std::endl;

  const string modelPath =
      "/home/xyz/projects/1690/model_trans/YOLOv5/models/BM1690/"
      "yolov5s_v6.1_3output_int8_1b.bmodel";
  tpuRtStatus_t status;
  tpuRtStream_t stream;
  // init device
  tpuRtInit();
  tpuRtSetDevice(0);
  tpuRtNetContext_t context;
  status = tpuRtCreateNetContext(&context);
  tpuRtStreamCreate(&stream);
  tpuRtNet_t net;
  tpuRtNetInfo_t info;
  // load bmodel
  status = tpuRtLoadNet(modelPath.c_str(), context, &net);
  info = tpuRtGetNetInfo(&net);

  tpuRtTensor_t* input_tensor;
  tpuRtTensor_t* output_tensor;
  input_tensor = (tpuRtTensor_t*)malloc(sizeof(tpuRtTensor_t) * info.input.num);
  output_tensor =
      (tpuRtTensor_t*)malloc(sizeof(tpuRtTensor_t) * info.output.num);

  for (int i = 0; i < info.input.num; i++) {
    input_tensor[i].dtype = info.input.dtypes[i];
    input_tensor[i].shape.num_dims = info.stages[0].input_shapes[i].num_dims;
    for (int k = 0; k < info.stages[0].input_shapes[0].num_dims; ++k) {
      input_tensor[i].shape.dims[k] = info.stages[0].input_shapes[i].dims[k];
    }
    int size = getTensorBytes(input_tensor[i]);
    tpuRtMalloc(&input_tensor[i].data, size, 0);
    tpuRtMemcpyS2D(input_tensor[i].data, inBuffer, size);
  }

  for (int i = 0; i < info.output.num; i++) {
    output_tensor[i].dtype = info.output.dtypes[i];
    output_tensor[i].shape.num_dims = info.stages[0].output_shapes[i].num_dims;
    for (int k = 0; k < info.stages[0].output_shapes[0].num_dims; ++k) {
      output_tensor[i].shape.dims[k] = info.stages[0].output_shapes[i].dims[k];
    }
    int size = getTensorBytes(output_tensor[i]);
    tpuRtMalloc(&output_tensor[i].data, size, 0);
  }
  status =
      tpuRtLaunchNetAsync(&net, input_tensor, output_tensor, info.name, stream);
  tpuRtStreamSynchronize(stream);
  for (int i = 0; i < info.output.num; i++) {
    tpuRtMemcpyD2S(outBuffer[i], output_tensor[i].data,
                   getTensorBytes(output_tensor[i]));
  }

  std::string outfilename = "../output.tpuRt";
  std::ofstream of(outfilename, std::ios::binary | std::ios::ate | std::ios::out);
  for(int i = 0; i < info.output.num; ++i) {
    of.write(outBuffer[i], dims[i]);
  }
  of.close();


  auto diff = getDiff(outBuffer, fileOutBuffer, dims);
  std::cout << "diff is " << diff << std::endl;

  delete[] inBuffer;
  delete[] outBuffer;
  delete[] fileOutBuffer;
  free(input_tensor);
  input_tensor = nullptr;
  free(output_tensor);
  output_tensor = nullptr;

  return 0;
}