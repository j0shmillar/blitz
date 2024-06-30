#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include "model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define INPUT_SIZE {input_size}  
#define OUTPUT_SIZE {output_size} 
#define NUM_RUNS {num_inferences}
#define DELAY_BETWEEN_INFERENCES {inference_seconds}

namespace {
  constexpr int kTensorArenaSize = 10 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

int main() {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        fprintf(stderr, "Model provided is version %d not equal to supported version %d.\n", model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    tflite::MicroMutableOpResolver<10> resolver; 
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    struct timeval start_time, end_time;
    double latencies[NUM_RUNS];

    for (int i = 0; i < NUM_RUNS; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            input->data.f[j] = (float)rand() / RAND_MAX;
        }

        gettimeofday(&start_time, NULL);
        TfLiteStatus invoke_status = interpreter.Invoke();
        gettimeofday(&end_time, NULL);

        if (invoke_status != kTfLiteOk) {
            fprintf(stderr, "Invoke failed\n");
            return 1;
        }

        double latency_ms = (double)(end_time.tv_sec - start_time.tv_sec) * 1000.0 + (double)(end_time.tv_usec - start_time.tv_usec) / 1000.0;
        latencies[i] = latency_ms;

        usleep((useconds_t)(DELAY_BETWEEN_INFERENCES * 1000000)); 
    }

    double total_time = 0.0;
    for (int i = 0; i < NUM_RUNS; ++i) {
        total_time += latencies[i];
    }
    double average_latency = total_time / NUM_RUNS;
    double throughput = NUM_RUNS / (total_time / 1000.0);

    printf("Average Latency: %.2f ms\n", average_latency);
    printf("Throughput: %.2f inferences/second\n", throughput);

    return 0;
}