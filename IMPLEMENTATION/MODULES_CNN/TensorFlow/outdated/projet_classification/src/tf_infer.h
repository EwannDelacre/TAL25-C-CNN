#ifndef TF_INFER_H
#define TF_INFER_H

#include <tensorflow/c/c_api.h>

TF_Graph* load_graph(const char *model_path);
TF_Session* create_session(TF_Graph *graph);
TF_Tensor* make_input_tensor(const float *data, int batch, int size);
TF_Tensor* run_inference(TF_Session *session, TF_Graph *graph, TF_Tensor *input, const char *input_op, const char *output_op);

void print_tensor_float(TF_Tensor *tensor);

#endif
