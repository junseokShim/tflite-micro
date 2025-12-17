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
#include "tensorflow/lite/kernels/internal/reference/gelu.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/gelu.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

void* GeluInit(TfLiteContext* context, const char* buffer,
               size_t length) {                                 // todo
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);  // nullptr check
  return context->AllocatePersistentBuffer(context, sizeof(GeluOpData));
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {  // todo
  return CalculateOpDataLeakyRelu(context, node);
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {  // todo
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Gelu(tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<float>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TFLMRegistration Register_GELU() {
  return tflite::micro::RegisterOp(tflite::GeluInit, tflite::GeluPrepare,
                                   tflite::GeluEval);
}

}  // namespace tflite