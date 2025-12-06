/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_GELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_GELU_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

inline void Gelu(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float kGamma = 0.044715f;

  for (int i = 0; i < flat_size; ++i) {
    const float x = input_data[i];

    const float x3 = x * x * x;
    const float arg = kAlpha * (x + kGamma * x3);
    const float th = Tanh(arg);  // 여기서 LUT 기반 Tanh 사용

    output_data[i] = 0.5f * x * (1.0f + th);
  }
}

}

#endif