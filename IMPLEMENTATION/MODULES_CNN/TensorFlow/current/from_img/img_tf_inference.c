/*
Programme de reconnaissance de chiffres MNIST avec TensorFlow C API
=====================================================================

Ce programme permet de charger une image de chiffre manuscrit au format MNIST (28x28 pixels)
et d'effectuer une inférence pour déterminer le chiffre représenté (0-9).

Remarques pour la compilation:

Ce code nécessite l'installation de TensorFlow C API ainsi que les libraires STB.
Assurez vous de compiler avec les bons flags :

gcc -I/usr/local/lib/tensorflow/include -L/usr/local/lib/tensorflow/lib -ltensorflow img_tf_inference.c -o img_tf_inference -lm

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

// Dimensions MNIST
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define NUM_CLASSES 10

// verif TF
void CheckStatus(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error: %s\n", TF_Message(status));
        exit(1);
    }
}

// free tenseur
void FreeTensor(void* data, size_t len, void* arg) {
    free(data);
}

// inférence sur l'image
void infer_mnist_image(float* image_data, int width, int height, TF_Graph* graph, TF_Session* session, TF_Status* status) {

    // prépa E/S
    TF_Output input_output;
    TF_Output softmax_output;
    TF_Tensor* input_values[1] = {NULL};
    TF_Tensor* output_values[1] = {NULL};

    input_output.oper = TF_GraphOperationByName(graph, "input");
    input_output.index = 0;
    softmax_output.oper = TF_GraphOperationByName(graph, "softmax");
    softmax_output.index = 0;

    if (!input_output.oper || !softmax_output.oper) {
        fprintf(stderr, "Erreur: Impossible de trouver les opérations dans le graphe\n");
        return;
    }

    TF_Output inputs[1] = {input_output};
    TF_Output outputs[1] = {softmax_output};

    int64_t input_dims[4] = {1, height, width, 1};
    size_t input_size = height * width * sizeof(float);
    float* input_data_copy = (float*)malloc(input_size);
    memcpy(input_data_copy, image_data, input_size);

    // création tenseur input
    input_values[0] = TF_NewTensor(TF_FLOAT, input_dims, 4, input_data_copy, input_size, FreeTensor, NULL);

    // exec session
    TF_SessionRun(
        session,
        NULL,
        inputs, input_values, 1,
        outputs, output_values, 1,
        NULL, 0,
        NULL,
        status
    );
    CheckStatus(status);

    // recup results
    float* softmax_result = (float*)TF_TensorData(output_values[0]);

    // print results
    int predicted_digit = 0;
    float max_prob = softmax_result[0];
    printf("Probabilités par classe:\n");

    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Chiffre %d: %.2f%%\n", i, softmax_result[i] * 100);

        if (softmax_result[i] > max_prob) {
            max_prob = softmax_result[i];
            predicted_digit = i;
        }
    }
    printf("\n=> Chiffre prédit: %d (confiance: %.2f%%)\n", predicted_digit, max_prob * 100);

    // free tenseurs
    TF_DeleteTensor(input_values[0]);
    TF_DeleteTensor(output_values[0]);
}

// fct de preprocess
void preprocess_image(unsigned char* image_data, int width, int height, int channels, float* processed_data) {

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float pixel = 0.0f;
            if (channels == 1) {
                pixel = image_data[i * width + j] / 255.0f;
            } else if (channels >= 3) {
                unsigned char* pixel_data = &image_data[(i * width + j) * channels];
                pixel = (pixel_data[0] + pixel_data[1] + pixel_data[2]) / (3.0f * 255.0f);
            }

            processed_data[i * width + j] = pixel;
        }
    }
}


void softmax(float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// fct TF pour construction de graphe
TF_Session* build_mnist_cnn_graph(TF_Graph** graph_out, TF_Status* status) {

    // init grraphe
    TF_Graph* graph = TF_NewGraph();
    *graph_out = graph;

    // init session
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    CheckStatus(status);

    // free session
    TF_DeleteSessionOptions(session_opts);

    int64_t input_dims[4] = {1, MNIST_HEIGHT, MNIST_WIDTH, 1};

    // input placeholder de lim'age'
    TF_OperationDescription* input_desc = TF_NewOperation(graph, "Placeholder", "input");
    TF_SetAttrType(input_desc, "dtype", TF_FLOAT);
    TF_SetAttrShape(input_desc, "shape", input_dims, 4);
    TF_Operation* input_op = TF_FinishOperation(input_desc, status);
    CheckStatus(status);

    // init tenseur de poids (16 filtres de 3x3)
    const int num_filters_layer1 = 16;
    float* kernel_data_layer1 = (float*)malloc(3 * 3 * 1 * num_filters_layer1 * sizeof(float));

    for (int i = 0; i < 3 * 3 * 1 * num_filters_layer1; i++) {
        // random simple init pour l'exemple
        kernel_data_layer1[i] = (float)(i % 9 - 4) / 10.0f;
    }

    // [height, width, in_channels, out_channels]
    int64_t kernel_dims_layer1[4] = {3, 3, 1, num_filters_layer1};

    // init const tenseur
    TF_OperationDescription* kernel_desc_layer1 = TF_NewOperation(graph, "Const", "conv1_kernel");
    TF_SetAttrType(kernel_desc_layer1, "dtype", TF_FLOAT);

    TF_Tensor* kernel_tensor_layer1 = TF_NewTensor(TF_FLOAT, kernel_dims_layer1, 4, kernel_data_layer1,
                                                  3 * 3 * 1 * num_filters_layer1 * sizeof(float),
                                                  FreeTensor, NULL);
    TF_SetAttrTensor(kernel_desc_layer1, "value", kernel_tensor_layer1, status);
    CheckStatus(status);

    TF_Operation* kernel_op_layer1 = TF_FinishOperation(kernel_desc_layer1, status);
    CheckStatus(status);

    // init tenseur pour les biais
    float* bias_data_layer1 = (float*)malloc(num_filters_layer1 * sizeof(float));
    for (int i = 0; i < num_filters_layer1; i++) {
        bias_data_layer1[i] = 0.1f;
    }
    int64_t bias_dims_layer1[1] = {num_filters_layer1};

    TF_OperationDescription* bias_desc_layer1 = TF_NewOperation(graph, "Const", "conv1_bias");
    TF_SetAttrType(bias_desc_layer1, "dtype", TF_FLOAT);

    TF_Tensor* bias_tensor_layer1 = TF_NewTensor(TF_FLOAT, bias_dims_layer1, 1, bias_data_layer1,
                                                num_filters_layer1 * sizeof(float),
                                                FreeTensor, NULL);
    TF_SetAttrTensor(bias_desc_layer1, "value", bias_tensor_layer1, status);
    CheckStatus(status);

    TF_Operation* bias_op_layer1 = TF_FinishOperation(bias_desc_layer1, status);
    CheckStatus(status);

    // Opération de convolution 2D
    TF_OperationDescription* conv_desc_layer1 = TF_NewOperation(graph, "Conv2D", "conv1");
    TF_Output input_op_out;
    input_op_out.oper = input_op;
    input_op_out.index = 0;
    TF_AddInput(conv_desc_layer1, input_op_out);

    TF_Output kernel_op_out_layer1;
    kernel_op_out_layer1.oper = kernel_op_layer1;
    kernel_op_out_layer1.index = 0;
    TF_AddInput(conv_desc_layer1, kernel_op_out_layer1);

    TF_SetAttrType(conv_desc_layer1, "T", TF_FLOAT);
    TF_SetAttrString(conv_desc_layer1, "padding", "SAME", 4);

    int64_t strides[4] = {1, 1, 1, 1};
    TF_SetAttrIntList(conv_desc_layer1, "strides", strides, 4);

    TF_Operation* conv_op_layer1 = TF_FinishOperation(conv_desc_layer1, status);
    CheckStatus(status);

    // ajout biais
    TF_OperationDescription* bias_add_desc_layer1 = TF_NewOperation(graph, "BiasAdd", "bias_add1");

    TF_Output conv_op_out_layer1;
    conv_op_out_layer1.oper = conv_op_layer1;
    conv_op_out_layer1.index = 0;
    TF_AddInput(bias_add_desc_layer1, conv_op_out_layer1);

    TF_Output bias_op_out_layer1;
    bias_op_out_layer1.oper = bias_op_layer1;
    bias_op_out_layer1.index = 0;
    TF_AddInput(bias_add_desc_layer1, bias_op_out_layer1);

    TF_SetAttrType(bias_add_desc_layer1, "T", TF_FLOAT);

    TF_Operation* bias_add_op_layer1 = TF_FinishOperation(bias_add_desc_layer1, status);
    CheckStatus(status);

    // appel ReLU
    TF_OperationDescription* relu_desc_layer1 = TF_NewOperation(graph, "Relu", "relu1");

    TF_Output bias_add_op_out_layer1;
    bias_add_op_out_layer1.oper = bias_add_op_layer1;
    bias_add_op_out_layer1.index = 0;
    TF_AddInput(relu_desc_layer1, bias_add_op_out_layer1);

    TF_SetAttrType(relu_desc_layer1, "T", TF_FLOAT);

    TF_Operation* relu_op_layer1 = TF_FinishOperation(relu_desc_layer1, status);
    CheckStatus(status);

    TF_OperationDescription* pool_desc = TF_NewOperation(graph, "MaxPool", "pool1");

    TF_Output relu_op_out_layer1;
    relu_op_out_layer1.oper = relu_op_layer1;
    relu_op_out_layer1.index = 0;
    TF_AddInput(pool_desc, relu_op_out_layer1);

    int64_t ksize[4] = {1, 2, 2, 1};
    TF_SetAttrIntList(pool_desc, "ksize", ksize, 4);
    TF_SetAttrIntList(pool_desc, "strides", ksize, 4);
    TF_SetAttrString(pool_desc, "padding", "VALID", 5);
    TF_SetAttrType(pool_desc, "T", TF_FLOAT);

    TF_Operation* pool_op = TF_FinishOperation(pool_desc, status);
    CheckStatus(status);

    // 2e couche de convolution 3x3

    const int num_filters_layer2 = 32;
    float* kernel_data_layer2 = (float*)malloc(3 * 3 * num_filters_layer1 * num_filters_layer2 * sizeof(float));

    // init poids
    for (int i = 0; i < 3 * 3 * num_filters_layer1 * num_filters_layer2; i++) {
        kernel_data_layer2[i] = (float)(i % 9 - 4) / 20.0f;
    }

    int64_t kernel_dims_layer2[4] = {3, 3, num_filters_layer1, num_filters_layer2};

    TF_OperationDescription* kernel_desc_layer2 = TF_NewOperation(graph, "Const", "conv2_kernel");
    TF_SetAttrType(kernel_desc_layer2, "dtype", TF_FLOAT);

    TF_Tensor* kernel_tensor_layer2 = TF_NewTensor(TF_FLOAT, kernel_dims_layer2, 4, kernel_data_layer2,
                                                 3 * 3 * num_filters_layer1 * num_filters_layer2 * sizeof(float),
                                                 FreeTensor, NULL);
    TF_SetAttrTensor(kernel_desc_layer2, "value", kernel_tensor_layer2, status);
    CheckStatus(status);

    TF_Operation* kernel_op_layer2 = TF_FinishOperation(kernel_desc_layer2, status);
    CheckStatus(status);

    // biais de la 2e couche
    float* bias_data_layer2 = (float*)malloc(num_filters_layer2 * sizeof(float));
    for (int i = 0; i < num_filters_layer2; i++) {
        bias_data_layer2[i] = 0.1f;
    }
    int64_t bias_dims_layer2[1] = {num_filters_layer2};

    TF_OperationDescription* bias_desc_layer2 = TF_NewOperation(graph, "Const", "conv2_bias");
    TF_SetAttrType(bias_desc_layer2, "dtype", TF_FLOAT);

    TF_Tensor* bias_tensor_layer2 = TF_NewTensor(TF_FLOAT, bias_dims_layer2, 1, bias_data_layer2,
                                               num_filters_layer2 * sizeof(float),
                                               FreeTensor, NULL);
    TF_SetAttrTensor(bias_desc_layer2, "value", bias_tensor_layer2, status);
    CheckStatus(status);

    TF_Operation* bias_op_layer2 = TF_FinishOperation(bias_desc_layer2, status);
    CheckStatus(status);

    // convolution 2D (2e couche)
    TF_OperationDescription* conv_desc_layer2 = TF_NewOperation(graph, "Conv2D", "conv2");

    TF_Output pool_op_out;
    pool_op_out.oper = pool_op;
    pool_op_out.index = 0;
    TF_AddInput(conv_desc_layer2, pool_op_out);

    TF_Output kernel_op_out_layer2;
    kernel_op_out_layer2.oper = kernel_op_layer2;
    kernel_op_out_layer2.index = 0;
    TF_AddInput(conv_desc_layer2, kernel_op_out_layer2);

    TF_SetAttrType(conv_desc_layer2, "T", TF_FLOAT);
    TF_SetAttrString(conv_desc_layer2, "padding", "SAME", 4);
    TF_SetAttrIntList(conv_desc_layer2, "strides", strides, 4);

    TF_Operation* conv_op_layer2 = TF_FinishOperation(conv_desc_layer2, status);
    CheckStatus(status);

    // ajout biais (2e couche)
    TF_OperationDescription* bias_add_desc_layer2 = TF_NewOperation(graph, "BiasAdd", "bias_add2");

    TF_Output conv_op_out_layer2;
    conv_op_out_layer2.oper = conv_op_layer2;
    conv_op_out_layer2.index = 0;
    TF_AddInput(bias_add_desc_layer2, conv_op_out_layer2);

    TF_Output bias_op_out_layer2;
    bias_op_out_layer2.oper = bias_op_layer2;
    bias_op_out_layer2.index = 0;
    TF_AddInput(bias_add_desc_layer2, bias_op_out_layer2);

    TF_SetAttrType(bias_add_desc_layer2, "T", TF_FLOAT);

    TF_Operation* bias_add_op_layer2 = TF_FinishOperation(bias_add_desc_layer2, status);
    CheckStatus(status);

    // appel ReLU (2e couche)
    TF_OperationDescription* relu_desc_layer2 = TF_NewOperation(graph, "Relu", "relu2");

    TF_Output bias_add_op_out_layer2;
    bias_add_op_out_layer2.oper = bias_add_op_layer2;
    bias_add_op_out_layer2.index = 0;
    TF_AddInput(relu_desc_layer2, bias_add_op_out_layer2);

    TF_SetAttrType(relu_desc_layer2, "T", TF_FLOAT);

    TF_Operation* relu_op_layer2 = TF_FinishOperation(relu_desc_layer2, status);
    CheckStatus(status);

    // GAP (2e couche)
    TF_OperationDescription* pool_desc2 = TF_NewOperation(graph, "MaxPool", "pool2");

    TF_Output relu_op_out_layer2;
    relu_op_out_layer2.oper = relu_op_layer2;
    relu_op_out_layer2.index = 0;
    TF_AddInput(pool_desc2, relu_op_out_layer2);

    TF_SetAttrIntList(pool_desc2, "ksize", ksize, 4);
    TF_SetAttrIntList(pool_desc2, "strides", ksize, 4);
    TF_SetAttrString(pool_desc2, "padding", "VALID", 5);
    TF_SetAttrType(pool_desc2, "T", TF_FLOAT);

    TF_Operation* pool_op2 = TF_FinishOperation(pool_desc2, status);
    CheckStatus(status);


    //  2 poolings en 2x2 donc 28/2/2 = 7, donc 7x7x32 = 1568 éléments
    int flatten_size = 7 * 7 * num_filters_layer2;

    // init tenseur reshape
    int64_t shape_dims[1] = {2};
    int* shape_data = (int*)malloc(shape_dims[0] * sizeof(int));
    shape_data[0] = 1;
    shape_data[1] = flatten_size;

    TF_OperationDescription* shape_desc = TF_NewOperation(graph, "Const", "flatten_shape");
    TF_SetAttrType(shape_desc, "dtype", TF_INT32);

    TF_Tensor* shape_tensor = TF_NewTensor(TF_INT32, shape_dims, 1, shape_data,
                                          shape_dims[0] * sizeof(int),
                                          FreeTensor, NULL);
    TF_SetAttrTensor(shape_desc, "value", shape_tensor, status);
    CheckStatus(status);

    TF_Operation* shape_op = TF_FinishOperation(shape_desc, status);
    CheckStatus(status);

    // reshape ops
    TF_OperationDescription* reshape_desc = TF_NewOperation(graph, "Reshape", "flatten");

    TF_Output pool_op_out2;
    pool_op_out2.oper = pool_op2;
    pool_op_out2.index = 0;
    TF_AddInput(reshape_desc, pool_op_out2);

    TF_Output shape_op_out;
    shape_op_out.oper = shape_op;
    shape_op_out.index = 0;
    TF_AddInput(reshape_desc, shape_op_out);

    TF_SetAttrType(reshape_desc, "T", TF_FLOAT);
    TF_SetAttrType(reshape_desc, "Tshape", TF_INT32);

    TF_Operation* reshape_op = TF_FinishOperation(reshape_desc, status);
    CheckStatus(status);

    // derniere couche

    // poids derniere couche
    float* fc_weights_data = (float*)malloc(flatten_size * NUM_CLASSES * sizeof(float));
    for (int i = 0; i < flatten_size * NUM_CLASSES; i++) {
        fc_weights_data[i] = (float)(i % 10 - 5) / (10.0f * flatten_size);
    }
    int64_t fc_weights_dims[2] = {flatten_size, NUM_CLASSES};

    TF_OperationDescription* fc_weights_desc = TF_NewOperation(graph, "Const", "fc_weights");
    TF_SetAttrType(fc_weights_desc, "dtype", TF_FLOAT);

    TF_Tensor* fc_weights_tensor = TF_NewTensor(TF_FLOAT, fc_weights_dims, 2, fc_weights_data,
                                               flatten_size * NUM_CLASSES * sizeof(float),
                                               FreeTensor, NULL);
    TF_SetAttrTensor(fc_weights_desc, "value", fc_weights_tensor, status);
    CheckStatus(status);

    TF_Operation* fc_weights_op = TF_FinishOperation(fc_weights_desc, status);
    CheckStatus(status);

    // biais dernier couche
    float* fc_bias_data = (float*)malloc(NUM_CLASSES * sizeof(float));
    for (int i = 0; i < NUM_CLASSES; i++) {
        fc_bias_data[i] = 0.1f;
    }
    int64_t fc_bias_dims[1] = {NUM_CLASSES};

    TF_OperationDescription* fc_bias_desc = TF_NewOperation(graph, "Const", "fc_bias");
    TF_SetAttrType(fc_bias_desc, "dtype", TF_FLOAT);

    TF_Tensor* fc_bias_tensor = TF_NewTensor(TF_FLOAT, fc_bias_dims, 1, fc_bias_data,
                                            NUM_CLASSES * sizeof(float),
                                            FreeTensor, NULL);
    TF_SetAttrTensor(fc_bias_desc, "value", fc_bias_tensor, status);
    CheckStatus(status);

    TF_Operation* fc_bias_op = TF_FinishOperation(fc_bias_desc, status);
    CheckStatus(status);

    // mult matrice
    TF_OperationDescription* matmul_desc = TF_NewOperation(graph, "MatMul", "fc_matmul");

    TF_Output reshape_op_out;
    reshape_op_out.oper = reshape_op;
    reshape_op_out.index = 0;
    TF_AddInput(matmul_desc, reshape_op_out);

    TF_Output fc_weights_op_out;
    fc_weights_op_out.oper = fc_weights_op;
    fc_weights_op_out.index = 0;
    TF_AddInput(matmul_desc, fc_weights_op_out);

    TF_SetAttrType(matmul_desc, "T", TF_FLOAT);
    TF_SetAttrBool(matmul_desc, "transpose_a", 0);
    TF_SetAttrBool(matmul_desc, "transpose_b", 0);

    TF_Operation* matmul_op = TF_FinishOperation(matmul_desc, status);
    CheckStatus(status);

    // ajout biais
    TF_OperationDescription* fc_bias_add_desc = TF_NewOperation(graph, "BiasAdd", "fc_bias_add");

    TF_Output matmul_op_out;
    matmul_op_out.oper = matmul_op;
    matmul_op_out.index = 0;
    TF_AddInput(fc_bias_add_desc, matmul_op_out);

    TF_Output fc_bias_op_out;
    fc_bias_op_out.oper = fc_bias_op;
    fc_bias_op_out.index = 0;
    TF_AddInput(fc_bias_add_desc, fc_bias_op_out);

    TF_SetAttrType(fc_bias_add_desc, "T", TF_FLOAT);

    TF_Operation* fc_bias_add_op = TF_FinishOperation(fc_bias_add_desc, status);
    CheckStatus(status);

    // softmax
    TF_OperationDescription* softmax_desc = TF_NewOperation(graph, "Softmax", "softmax");

    TF_Output fc_bias_add_op_out;
    fc_bias_add_op_out.oper = fc_bias_add_op;
    fc_bias_add_op_out.index = 0;
    TF_AddInput(softmax_desc, fc_bias_add_op_out);

    TF_SetAttrType(softmax_desc, "T", TF_FLOAT);

    TF_Operation* softmax_op = TF_FinishOperation(softmax_desc, status);
    CheckStatus(status);

    // Libérer la mémoire des tenseurs constants
    TF_DeleteTensor(kernel_tensor_layer1);
    TF_DeleteTensor(bias_tensor_layer1);
    TF_DeleteTensor(kernel_tensor_layer2);
    TF_DeleteTensor(bias_tensor_layer2);
    TF_DeleteTensor(shape_tensor);
    TF_DeleteTensor(fc_weights_tensor);
    TF_DeleteTensor(fc_bias_tensor);

    return session;
}

// fct main
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    // chargement modèle
    TF_Graph* graph = load_model("model.tflite");
    if (!graph) {
        fprintf(stderr, "Échec du chargement du modèle\n");
        return 1;
    }

    // chargement image input
    const char* image_path = argv[1];
    int width, height, channels;
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 0);
    if (!image_data) {
        fprintf(stderr, "Impossible de charger l'image %s\n", image_path);
        TF_DeleteGraph(graph);
        return 1;
    }

    printf("Image chargée: %s (%dx%d avec %d canaux)\n", image_path, width, height, channels);

    float preprocessed_data[28 * 28];
    preprocess_image(image_data, width, height, channels, preprocessed_data);

    printf("Échantillon de l'image prétraitée (10 premiers pixels):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", preprocessed_data[i]);
    }
    printf("...\n");

    // print terminal de l'image
    printf("Représentation visuelle de l'image:\n");
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            float pixel_val = preprocessed_data[i * 28 + j];
            if (pixel_val > 0.5f) {
                printf("##");
            } else if (pixel_val > 0.1f) {
                printf("# ");
            } else {
                printf("  ");
            }
        }
        printf("\n");
    }

    // init ession TF
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Erreur lors de la création de la session: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(session_opts);
        TF_DeleteGraph(graph);
        stbi_image_free(image_data);
        return 1;
    }

    // prépa tenseurs d'E/S
    TF_Output input_op = {TF_GraphOperationByName(graph, "input_1"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "dense_2/Softmax"), 0};

    // verifs ops
    if (!input_op.oper || !output_op.oper) {
        fprintf(stderr, "Erreur: Impossible de trouver les opérations d'entrée/sortie dans le graphe\n");
        TF_DeleteSession(session, status);
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(session_opts);
        TF_DeleteGraph(graph);
        stbi_image_free(image_data);
        return 1;
    }

    // création tensuer input
    int64_t input_dims[4] = {1, 28, 28, 1};
    size_t input_size = 1 * 28 * 28 * 1 * sizeof(float);

    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, input_dims, 4, input_size);
    float* input_data = (float*)TF_TensorData(input_tensor);

    memcpy(input_data, preprocessed_data, 28 * 28 * sizeof(float));

    TF_Tensor* input_tensors[] = {input_tensor};
    TF_Tensor* output_tensors[1] = {NULL};

    // exec  inférence
    TF_SessionRun(
        session,
        NULL,
        &input_op, input_tensors, 1,
        &output_op, output_tensors, 1,
        NULL, 0,
        NULL,
        status
    );

    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Erreur lors de l'inférence: %s\n", TF_Message(status));
        TF_DeleteTensor(input_tensor);
        TF_DeleteSession(session, status);
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(session_opts);
        TF_DeleteGraph(graph);
        stbi_image_free(image_data);
        return 1;
    }

    float* output_data = TF_TensorData(output_tensors[0]);
    int output_size = 10;  // Pour MNIST (10 chiffres)

    // appel softmax
    float probabilities[10];
    if (strstr(TF_OperationName(output_op.oper), "Softmax") != NULL) {
        memcpy(probabilities, output_data, output_size * sizeof(float));
    } else {
        softmax(output_data, probabilities, output_size);
    }

    // print resultat
    float max_prob = 0.0f;
    int predicted_class = -1;

    printf("Probabilités par classe:\n");
    for (int i = 0; i < output_size; i++) {
        printf("Chiffre %d: %.2f%%\n", i, probabilities[i] * 100);
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            predicted_class = i;
        }
    }

    printf("=> Chiffre prédit: %d (confiance: %.2f%%)\n", predicted_class, max_prob * 100);

    // Libérer les ressources
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensors[0]);
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    stbi_image_free(image_data);

    return 0;
}
