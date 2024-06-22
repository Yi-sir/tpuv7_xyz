#include <math.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "tpu_utils.h"

struct YoloV5Box {
  int x, y, width, height;
  float score;
  int class_id;
};

template <class T>
struct Point {
  Point() : mX(0), mY(0) {}

  Point(T x, T y) : mX(x), mY(y) {}

  T mX;
  T mY;
};

template <class T>
struct Rectangle {
  Rectangle() : mX(0), mY(0), mWidth(0), mHeight(0) {}

  Rectangle(T x, T y, T width, T height)
      : mX(x), mY(y), mWidth(width), mHeight(height) {}

  T top() const { return mY; }

  T bottom() const { return mY + mHeight; }

  T left() const { return mX; }

  T right() const { return mX + mWidth; }

  Point<T> center() const {
    return Point<T>(mX + mWidth / 2, mY + mHeight / 2);
  }

  T area() const { return mWidth * mHeight; }

  bool empty() const { return 0 == mWidth || 0 == mHeight; }

  T mX;
  T mY;
  T mWidth;
  T mHeight;
};

using YoloV5BoxVec = std::vector<YoloV5Box>;

void NMS(YoloV5BoxVec& dets, float nmsConfidence) {
  int length = dets.size();
  int index = length - 1;

  std::sort(
      dets.begin(), dets.end(),
      [](const YoloV5Box& a, const YoloV5Box& b) { return a.score < b.score; });

  std::vector<float> areas(length);
  for (int i = 0; i < length; i++) {
    areas[i] = dets[i].width * dets[i].height;
  }

  while (index > 0) {
    int i = 0;
    while (i < index) {
      float left = std::max(dets[index].x, dets[i].x);
      float top = std::max(dets[index].y, dets[i].y);
      float right = std::min(dets[index].x + dets[index].width,
                             dets[i].x + dets[i].width);
      float bottom = std::min(dets[index].y + dets[index].height,
                              dets[i].y + dets[i].height);
      float overlap =
          std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
      if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
        areas.erase(areas.begin() + i);
        dets.erase(dets.begin() + i);
        index--;
      } else {
        i++;
      }
    }
    index--;
  }
}

float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                              bool* pIsAligWidth) {
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w) {
    *pIsAligWidth = true;
    ratio = r_w;
  } else {
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

int argmax(float* data, int num) {
  float max_value = 0.0;
  int max_index = 0;
  for (int i = 0; i < num; ++i) {
    float value = data[i];
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  return max_index;
}

float sigmoid(float x) { return 1.0 / (1 + expf(-x)); }

struct PointMetadata {
  int getLabel() const {
    if (mTopKLabels.empty()) {
      return -1;
    } else {
      return mTopKLabels.front();
    }
  }

  float getScore() const {
    int label = getLabel();
    if (label < 0 || label >= mScores.size()) {
      return 0.f;
    } else {
      return mScores[label];
    }
  }

  Point<int> mPoint;
  std::vector<float> mScores;
  std::vector<int> mTopKLabels;
};

struct DetectedObjectMetadata {
  DetectedObjectMetadata() : mClassify(-1), mTrackIouThreshold(0.f) {}

  int getLabel() const {
    if (mTopKLabels.empty()) {
      return -1;
    } else {
      return mTopKLabels.front();
    }
  }

  float getScore() const {
    int label = getLabel();
    if (label < 0 || label >= mScores.size()) {
      return 0.f;
    } else {
      return mScores[label];
    }
  }

  Rectangle<int> mBox;
  // mCroppedBox are the dilated boxes by using mBox
  Rectangle<int> mCroppedBox;
  std::string mItemName;
  std::string mLabelName;
  std::vector<float> mScores;
  std::vector<int> mTopKLabels;
  int mClassify;
  std::string mClassifyName;
  float mTrackIouThreshold;
  std::vector<std::shared_ptr<PointMetadata>> mKeyPoints;
};

std::vector<std::shared_ptr<DetectedObjectMetadata>> postProcessCPU(
    char** outBuffers,
    std::vector<std::shared_ptr<BMNNTensor>> outputBMNNTensors) {
  YoloV5BoxVec yolobox_vec;
  int idx = 0;
  yolobox_vec.clear();
  int frame_width = 1920;
  int frame_height = 1080;

  int tx1 = 0, ty1 = 0;
  bool isAlignWidth = false;
  float ratio = get_aspect_scaled_ratio(frame_width, frame_height, 640, 640,
                                        &isAlignWidth);
  if (isAlignWidth) {
    ty1 = (int)((640 - (int)((frame_height)*ratio)) / 2);
  } else {
    tx1 = (int)((640 - (int)((frame_width)*ratio)) / 2);
  }
  int min_idx = 0;
  int box_num = 0;
  int min_dim = 9999;
  for (int i = 0; i < 3; ++i) {
    auto output_shape = outputBMNNTensors[i]->get_shape();
    auto output_dims = output_shape->num_dims;
    if (output_dims == 5) {
      box_num +=
          output_shape->dims[1] * output_shape->dims[2] * output_shape->dims[3];
    }

    if (min_dim > output_dims) {
      min_idx = i;
      min_dim = output_dims;
    }
  }

  auto out_tensor = outputBMNNTensors[min_idx];
  int nout = out_tensor->get_shape()->dims[min_dim - 1];
  int m_class_num = nout - 5;

  int out_nout = 7;
  int max_wh = 7680;
  bool agnostic = false;

  float* output_data = nullptr;
  std::vector<float> decoded_data;

  if (min_dim == 5) {
    const std::vector<std::vector<std::vector<int>>> anchors{
        {{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}};
    const int anchor_num = anchors[0].size();
    if ((int)decoded_data.size() != box_num * out_nout) {
      decoded_data.resize(box_num * out_nout);
    }
    float* dst = decoded_data.data();

    for (int tidx = 0; tidx < 3; ++tidx) {
      auto output_tensor = outputBMNNTensors[tidx];
      int feat_c = output_tensor->get_shape()->dims[1];
      int feat_h = output_tensor->get_shape()->dims[2];
      int feat_w = output_tensor->get_shape()->dims[3];
      int area = feat_h * feat_w;
      int feature_size = feat_h * feat_w * nout;
      float* tensor_data = reinterpret_cast<float*>(outBuffers[tidx]);

      for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
        float* ptr = tensor_data + anchor_idx * feature_size;
        for (int i = 0; i < area; i++) {
          if (ptr[4] > 0.5) {
            dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * 640;
            dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * 640;
            dst[2] = std::pow((sigmoid(ptr[2]) * 2), 2) *
                     anchors[tidx][anchor_idx][0];
            dst[3] = std::pow((sigmoid(ptr[3]) * 2), 2) *
                     anchors[tidx][anchor_idx][1];
            dst[4] = sigmoid(ptr[4]);

            dst[5] = ptr[5];
            dst[6] = 5;
            for (int d = 6; d < nout; d++) {
              if (ptr[d] > dst[5]) {
                dst[5] = ptr[d];
                dst[6] = d;
              }
            }
            dst[6] -= 5;
            float score = dst[4];

            int class_id = dst[6];
            float confidence = dst[5];
            float cur_class_thresh = 0.5;
            float box_transformed_m_conf_threshold =
                -std::log(score / cur_class_thresh - 1);
            if (confidence > box_transformed_m_conf_threshold) {
              float centerX = dst[0];
              float centerY = dst[1];
              float width = dst[2];
              float height = dst[3];

              YoloV5Box box;
              if (!agnostic)
                box.x = centerX - width / 2 + class_id * max_wh;
              else
                box.x = centerX - width / 2;
              if (box.x < 0) box.x = 0;
              if (!agnostic)
                box.y = centerY - height / 2 + class_id * max_wh;
              else
                box.y = centerY - height / 2;
              if (box.y < 0) box.y = 0;
              box.width = width;
              box.height = height;
              box.class_id = class_id;
              confidence = sigmoid(confidence);
              box.score = confidence * score;
              yolobox_vec.push_back(box);
            }
          }
          dst += out_nout;
          ptr += nout;
        }
      }
    }
    output_data = decoded_data.data();
  }

  NMS(yolobox_vec, 0.5);

  if (!agnostic)
    for (auto& box : yolobox_vec) {
      box.x -= box.class_id * max_wh;
      box.y -= box.class_id * max_wh;
      box.x = (box.x - tx1) / ratio;
      if (box.x < 0) box.x = 0;
      box.y = (box.y - ty1) / ratio;
      if (box.y < 0) box.y = 0;
      box.width = (box.width) / ratio;
      if (box.x + box.width >= frame_width) box.width = frame_width - box.x;
      box.height = (box.height) / ratio;
      if (box.y + box.height >= frame_height) box.height = frame_height - box.y;
    }
  else
    for (auto& box : yolobox_vec) {
      box.x = (box.x - tx1) / ratio;
      if (box.x < 0) box.x = 0;
      box.y = (box.y - ty1) / ratio;
      if (box.y < 0) box.y = 0;
      box.width = (box.width) / ratio;
      if (box.x + box.width >= frame_width) box.width = frame_width - box.x;
      box.height = (box.height) / ratio;
      if (box.y + box.height >= frame_height) box.height = frame_height - box.y;
    }
  std::vector<std::shared_ptr<DetectedObjectMetadata>> detDatas;
  for (auto bbox : yolobox_vec) {
    std::shared_ptr<DetectedObjectMetadata> detData =
        std::make_shared<DetectedObjectMetadata>();
    detData->mBox.mX = bbox.x;
    detData->mBox.mY = bbox.y;
    detData->mBox.mWidth = bbox.width;
    detData->mBox.mHeight = bbox.height;
    detData->mScores.push_back(bbox.score);
    detData->mClassify = bbox.class_id;
    detDatas.push_back(detData);
  }
  return detDatas;
}