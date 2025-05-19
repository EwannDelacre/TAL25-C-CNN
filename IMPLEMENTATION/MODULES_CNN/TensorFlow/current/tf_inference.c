/*
Remarques pour la compilation:

Ce code nécessite l'installation de TensorFlow C API.
Assurez vous de compiler avec les bons flags :

gcc -Wall tf_inference.c -o tf_inference -I/usr/local/lib/tensorflow/include -L/usr/local/lib/tensorflow/lib -ltensorflow

*/


#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>

// verif TF
void CheckStatus(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error: %s\n", TF_Message(status));
        exit(1);
    }
}

// free tenseur
void FreeTensor(void* data, size_t len, void* arg) {
    // do nothing
}

int main() {
    printf("Version de TensorFlow : %s\n", TF_Version());

    // init TF
    TF_Status* status = TF_NewStatus();

    // init graphe
    TF_Graph* graph = TF_NewGraph();

    // init session
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    CheckStatus(status);

    // free session
    TF_DeleteSessionOptions(session_opts);

    // params input
    int64_t input_dims[4] = {1, 4, 4, 1};

    // image 2D 4x4
    float image_data[16] = {
        1, 2, 0, 1,
        3, 1, 2, 1,
        0, 1, 3, 2,
        1, 0, 2, 3
    };

    // input placeholder de l'image
    TF_OperationDescription* input_desc = TF_NewOperation(graph, "Placeholder", "input");
    TF_SetAttrType(input_desc, "dtype", TF_FLOAT);
    TF_SetAttrShape(input_desc, "shape", input_dims, 4);
    TF_Operation* input_op = TF_FinishOperation(input_desc, status);
    CheckStatus(status);


    // init tenseur de poids (3 filtres de 2x2)
    float kernel_data[12] = {
        // 1
        1, -1,
        0, 1,
        // 2
        0, 1,
        1, -1,
        // 3
        -1, 0,
        1, 1
    };
    int64_t kernel_dims[4] = {2, 2, 1, 3};  // [height, width, in_channels, out_channels]

    // init const tenseur
    TF_OperationDescription* kernel_desc = TF_NewOperation(graph, "Const", "conv_kernel");
    TF_SetAttrType(kernel_desc, "dtype", TF_FLOAT);

    // allocation mémoire kernel
    float* kernel_copy = (float*)malloc(sizeof(kernel_data));
    memcpy(kernel_copy, kernel_data, sizeof(kernel_data));

    TF_Tensor* kernel_tensor = TF_NewTensor(TF_FLOAT, kernel_dims, 4, kernel_copy, sizeof(kernel_data), FreeTensor, NULL);
    TF_SetAttrTensor(kernel_desc, "value", kernel_tensor, status);
    CheckStatus(status);

    TF_Operation* kernel_op = TF_FinishOperation(kernel_desc, status);
    CheckStatus(status);

    // init tenseur pour les biais (1 par filtre)
    float bias_data[3] = {0.5, -0.3, 0.1};
    int64_t bias_dims[1] = {3};

    // allocation mémoire biais
    float* bias_copy = (float*)malloc(sizeof(bias_data));
    memcpy(bias_copy, bias_data, sizeof(bias_data));

    TF_OperationDescription* bias_desc = TF_NewOperation(graph, "Const", "conv_bias");
    TF_SetAttrType(bias_desc, "dtype", TF_FLOAT);

    TF_Tensor* bias_tensor = TF_NewTensor(TF_FLOAT, bias_dims, 1, bias_copy, sizeof(bias_data), FreeTensor, NULL);
    TF_SetAttrTensor(bias_desc, "value", bias_tensor, status);
    CheckStatus(status);

    TF_Operation* bias_op = TF_FinishOperation(bias_desc, status);
    CheckStatus(status);

    // convolution 2D
    TF_OperationDescription* conv_desc = TF_NewOperation(graph, "Conv2D", "conv");
    TF_Output input_op_out;
    input_op_out.oper = input_op;
    input_op_out.index = 0;
    TF_AddInput(conv_desc, input_op_out);

    TF_Output kernel_op_out;
    kernel_op_out.oper = kernel_op;
    kernel_op_out.index = 0;
    TF_AddInput(conv_desc, kernel_op_out);

    TF_SetAttrType(conv_desc, "T", TF_FLOAT);
    TF_SetAttrString(conv_desc, "padding", "VALID", 5);

    int64_t strides[4] = {1, 1, 1, 1};  // batch, height, width, channel
    TF_SetAttrIntList(conv_desc, "strides", strides, 4);

    TF_Operation* conv_op = TF_FinishOperation(conv_desc, status);
    CheckStatus(status);

    // ajout biais
    TF_OperationDescription* bias_add_desc = TF_NewOperation(graph, "BiasAdd", "bias_add");

    TF_Output conv_op_out;
    conv_op_out.oper = conv_op;
    conv_op_out.index = 0;
    TF_AddInput(bias_add_desc, conv_op_out);

    TF_Output bias_op_out;
    bias_op_out.oper = bias_op;
    bias_op_out.index = 0;
    TF_AddInput(bias_add_desc, bias_op_out);

    TF_SetAttrType(bias_add_desc, "T", TF_FLOAT);

    TF_Operation* bias_add_op = TF_FinishOperation(bias_add_desc, status);
    CheckStatus(status);

    // fct d'activation ReLU
    TF_OperationDescription* relu_desc = TF_NewOperation(graph, "Relu", "relu");

    TF_Output bias_add_op_out;
    bias_add_op_out.oper = bias_add_op;
    bias_add_op_out.index = 0;
    TF_AddInput(relu_desc, bias_add_op_out);

    TF_SetAttrType(relu_desc, "T", TF_FLOAT);

    TF_Operation* relu_op = TF_FinishOperation(relu_desc, status);
    CheckStatus(status);

    // GAP
    TF_OperationDescription* axes_desc = TF_NewOperation(graph, "Const", "pool_axes");
    TF_SetAttrType(axes_desc, "dtype", TF_INT32);

    int axes_data[2] = {1, 2};  // Axes 1 et 2 (height, width)
    int64_t axes_dims[1] = {2};

    // allocation mémoire des axes
    int* axes_copy = (int*)malloc(sizeof(axes_data));
    memcpy(axes_copy, axes_data, sizeof(axes_data));

    TF_Tensor* axes_tensor = TF_NewTensor(TF_INT32, axes_dims, 1, axes_copy, sizeof(axes_data), FreeTensor, NULL);
    TF_SetAttrTensor(axes_desc, "value", axes_tensor, status);

    TF_Operation* axes_op = TF_FinishOperation(axes_desc, status);
    CheckStatus(status);

    // calcul par GAP
    TF_OperationDescription* gap_desc = TF_NewOperation(graph, "Mean", "global_avg_pool");

    TF_Output relu_op_out;
    relu_op_out.oper = relu_op;
    relu_op_out.index = 0;
    TF_AddInput(gap_desc, relu_op_out);

    TF_Output axes_op_out;
    axes_op_out.oper = axes_op;
    axes_op_out.index = 0;
    TF_AddInput(gap_desc, axes_op_out);

    TF_SetAttrType(gap_desc, "T", TF_FLOAT);
    TF_SetAttrBool(gap_desc, "keep_dims", 0);  // Ne pas garder les dimensions

    TF_Operation* gap_op = TF_FinishOperation(gap_desc, status);
    CheckStatus(status);

    // softmax final
    TF_OperationDescription* softmax_desc = TF_NewOperation(graph, "Softmax", "softmax");

    TF_Output gap_op_out;
    gap_op_out.oper = gap_op;
    gap_op_out.index = 0;
    TF_AddInput(softmax_desc, gap_op_out);

    TF_SetAttrType(softmax_desc, "T", TF_FLOAT);

    TF_Operation* softmax_op = TF_FinishOperation(softmax_desc, status);
    CheckStatus(status);

    // input / output
    TF_Output input_output;
    input_output.oper = input_op;
    input_output.index = 0;

    TF_Output softmax_output;
    softmax_output.oper = softmax_op;
    softmax_output.index = 0;

    // init tenseurs input
    float* input_data_copy = (float*)malloc(sizeof(float) * 16);
    memcpy(input_data_copy, image_data, sizeof(float) * 16);

    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 4, input_data_copy, sizeof(float) * 16, FreeTensor, NULL);

    // stockage valeurs tenseur
    TF_Output inputs[1] = {input_output};
    TF_Tensor* input_values[1] = {input_tensor};

    TF_Output outputs[1] = {softmax_output};
    TF_Tensor* output_values[1] = {NULL};

    // exec session
    TF_SessionRun(
        session,
        NULL,                // Options de l'exécution
        inputs, input_values, 1,  // Entrées
        outputs, output_values, 1,  // Sorties
        NULL, 0,             // Opérations à exécuter (aucune supplémentaire)
        NULL,                // Metadata
        status
    );
    CheckStatus(status);

    // recup results
    float* softmax_result = (float*)TF_TensorData(output_values[0]);

    // print results
    printf("\nProbabilités (TensorFlow):\n");
    int predicted_class = 0;
    float max_prob = softmax_result[0];

    for (int i = 0; i < 3; i++) {
        printf("Classe %d : %.4f (%.2f%%)\n", i, softmax_result[i], softmax_result[i] * 100);

        if (softmax_result[i] > max_prob) {
            max_prob = softmax_result[i];
            predicted_class = i;
        }
    }

    printf("\n=> Classe prédite : %d\n", predicted_class);

    // clean
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(kernel_tensor);
    TF_DeleteTensor(bias_tensor);
    TF_DeleteTensor(axes_tensor);

    if (output_values[0] != NULL) {
        TF_DeleteTensor(output_values[0]);
    }

    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    // frr
    free(input_data_copy);
    free(kernel_copy);
    free(bias_copy);
    free(axes_copy);

    return 0;
}
