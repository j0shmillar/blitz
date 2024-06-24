#include <zephyr/kernel.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model.h"

namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  constexpr int kTensorArenaSize = 2 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
} 

int main(void)
{
  model = tflite::GetModel(g_model);
  static tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TF_LITE_ENSURE_STATUS(interpreter->AllocateTensors());

  input = interpreter->input(0);

  float TWO_PI = 3.14159265359f * 2;
  int NUM_CYCLES = 1000000;
  int BATCH_SIZE = 1000;
  printf("Running inference for %d cycles, %d a batch\n", NUM_CYCLES, BATCH_SIZE);
  uint32_t start_time = k_uptime_get_32();
  uint32_t batch_start_time = start_time;
  for (int i = 0; i < NUM_CYCLES; i++)
  {
    float x = TWO_PI * (i % BATCH_SIZE) / BATCH_SIZE;
    int8_t x_quantized = x / input->params.scale + input->params.zero_point;
    input->data.int8[0] = x_quantized;
    TF_LITE_ENSURE_STATUS(interpreter->Invoke());
    if (i % BATCH_SIZE == 0 && i != 0)
    {
      uint32_t t = k_uptime_get_32();
      printf("%04d, %d\n", i / BATCH_SIZE, t - batch_start_time);
      batch_start_time = t;
    }
  }

  printf("Average time: %d\n", (k_uptime_get_32() - start_time) * BATCH_SIZE / NUM_CYCLES);
  return 0;

}
