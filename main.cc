#include <fstream>
#include <iostream>
#include <numeric>

#include "post_process.cc"

const std::string ref_in = "../data/1684x/input_int81b";
const std::string ref_out = "../data/1684x/output_int81b";
const std::string calc_out = "../data/1690/output_int81b";
const std::string modelPath =
    "/home/xyz/projects/1690/model_trans/YOLOv5/models/BM1690/"
    "yolov5s_v6.1_3output_int8_1b.bmodel";

char* inBuffer;
char** outBuffer;
char** fileOutBuffer;

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

void prepareHostTensorsFromFile(const std::string& ref_in,
                                const std::string& ref_out,
                                std::vector<int>& dims) {
  std::ifstream file1(ref_in, std::ios::binary | std::ios::ate);
  if (file1.is_open()) {
    std::streampos fileSize = file1.tellg();
    file1.seekg(0, std::ios::beg);
    inBuffer = new char[int(fileSize)];
    file1.read(inBuffer, int(fileSize));
    file1.close();
  } else {
    std::cerr << "无法打开文件" << std::endl;
  }

  std::ifstream file2(ref_out, std::ios::binary | std::ios::ate);
  if (file2.is_open()) {
    std::streampos fileSize = file2.tellg();

    fileOutBuffer[0] = new char[dims[0]];
    fileOutBuffer[1] = new char[dims[1]];
    fileOutBuffer[2] = new char[dims[2]];

    file2.seekg(0, std::ios::beg);
    file2.read(fileOutBuffer[0], dims[0]);
    file2.read(fileOutBuffer[1], dims[1]);
    file2.read(fileOutBuffer[2], dims[2]);

    file2.close();
  } else {
    std::cerr << "无法打开文件" << std::endl;
  }

  outBuffer[0] = new char[dims[0]];
  outBuffer[1] = new char[dims[1]];
  outBuffer[2] = new char[dims[2]];

  return;
}

void mallocAndCopyTpuRtTensors(
    std::shared_ptr<BMNNNetwork> net,
    std::vector<std::shared_ptr<tpuRtTensor_t>>& inputTensors,
    std::vector<std::shared_ptr<tpuRtTensor_t>>& outputTensors,
    char* inBuffer) {
  for (int i = 0; i < net->inputTensorNum(); ++i) {
    int size = getTensorBytes(*inputTensors[i]);
    tpuRtMalloc(&(inputTensors[i]->data), size, 0);
    tpuRtMemcpyS2D(inputTensors[i]->data, inBuffer, size);
  }
  for (int i = 0; i < net->outputTensorNum(); ++i) {
    int size = getTensorBytes(*outputTensors[i]);
    tpuRtMalloc(&(outputTensors[i]->data), size, 0);
  }
  return;
}


int main() {
  tpuRtInit();
  tpuRtSetDevice(0);
  tpuRtStatus_t ret;
  std::vector<int> dims;
  long inSize, outSize;
  auto context = std::make_shared<BMNNContext>(modelPath.c_str());
  auto network = context->network();
  outBuffer = new char*[network->outputTensorNum()];
  fileOutBuffer = new char*[network->outputTensorNum()];

  std::vector<std::shared_ptr<tpuRtTensor_t>> inputTensors(
      network->inputTensorNum());
  for (int i = 0; i < network->inputTensorNum(); ++i) {
    inputTensors[i] = network->inputTpuRtTensor(i);
  }
  std::vector<std::shared_ptr<tpuRtTensor_t>> outputTensors(
      network->outputTensorNum());
  for (int i = 0; i < network->outputTensorNum(); ++i) {
    outputTensors[i] = network->outputTpuRtTensor(i);
    dims.push_back(getTensorBytes(*outputTensors[i]));
  }

  prepareHostTensorsFromFile(ref_in, ref_out, dims);
  mallocAndCopyTpuRtTensors(network, inputTensors, outputTensors, inBuffer);

  ret = network->forward(inputTensors, outputTensors);

  std::vector<std::shared_ptr<BMNNTensor>> outputBMNNTensors;
  for (int i = 0; i < network->outputTensorNum(); ++i) {
    outputBMNNTensors.push_back(std::make_shared<BMNNTensor>(
        "", 1.0, outputTensors[i].get(), network->getStream()));
    outBuffer[i] = outputBMNNTensors[i]->get_host_data();
  }
  std::vector<std::shared_ptr<DetectedObjectMetadata>> detDatas =
      postProcessCPU(fileOutBuffer, outputBMNNTensors);
  // for (int i = 0; i < detDatas.size(); ++i) {
  //   std::cout << detDatas[i]->mBox.mX << " " << detDatas[i]->mBox.mY << " "
  //             << detDatas[i]->mBox.mWidth << " " << detDatas[i]->mBox.mHeight
  //             << std::endl;
  // }
  // auto diff = getDiff(outBuffer, fileOutBuffer, dims);

  // std::cout << "diff is " << diff << std::endl;
  delete[] inBuffer;
  for (int i = 0; i < network->inputTensorNum(); ++i) delete[] fileOutBuffer[i];
  delete[] fileOutBuffer;

  for (int i = 0; i < network->inputTensorNum(); ++i)
    tpuRtFree(&inputTensors[i]->data, 0);

  for (int i = 0; i < network->outputTensorNum(); ++i)
    tpuRtFree(&outputTensors[i]->data, 0);
  return 0;
}