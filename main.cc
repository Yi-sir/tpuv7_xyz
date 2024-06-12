#include "bmnn_utils1690.h"

int main() {
  const std::string modelPath =
      "/home/xyz/projects/1690/model_trans/YOLOv5/models/BM1690/"
      "yolov5s_v6.1_3output_int8_1b.bmodel";
  tpuRtInit();
  tpuRtSetDevice(0);
  auto tpu_ctx = std::make_shared<BMNNContext>(modelPath.c_str());
  auto tpu_net = tpu_ctx->network();
  return 0;
}