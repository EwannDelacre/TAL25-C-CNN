#ifndef INFER_H
#define INFER_H

#ifdef __cplusplus
extern "C" {
#endif

void setup_model(void);
void run_inference(const float *input, int input_size, float *output, int output_size);

#ifdef __cplusplus
}
#endif

#endif // INFER_H
