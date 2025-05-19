#include "tf_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// désallocation manuelle (pas tensorflow parce que sinon data de tensor plante)

void NoOpDeallocator(void* data, size_t a, void* b) {}

// on charge un graphe tensorflow depuis le .pb

TF_Graph* load_graph(const char *model_path) {
    TF_Graph *graph = TF_NewGraph(); //création graphe
    TF_Status *status = TF_NewStatus();

    TF_Buffer *buffer = TF_NewBuffer();
    FILE *f = fopen(model_path, "rb"); // ouverture du .pb
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(len);
    fread(data, 1, len, f); // lire contenu en binaire
    fclose(f);

    buffer->data = data;
    buffer->length = len;
    buffer->data_deallocator = free;

    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, opts, status); //charge dans le graphe
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    if (TF_GetCode(status) != TF_OK) {
        printf("ERROR: Unable to load graph: %s\n", TF_Message(status));
        exit(1);
    }

    TF_DeleteStatus(status);
    return graph;
}

// création d'un session d'inférence

TF_Session* create_session(TF_Graph *graph) {
    TF_Status *status = TF_NewStatus();
    TF_SessionOptions *opts = TF_NewSessionOptions();
    TF_Session *sess = TF_NewSession(graph, opts, status);
    if (TF_GetCode(status) != TF_OK) {
        printf("ERROR: Unable to create session: %s\n", TF_Message(status));
        exit(1);
    }
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(opts);
    return sess;
}

// création d'un tensor d'entrée

TF_Tensor* make_input_tensor(const float *data, int batch, int size) {
    int64_t dims[2] = { batch, size };
    size_t nbytes = batch * size * sizeof(float);
    TF_Tensor *tensor = TF_NewTensor(TF_FLOAT, dims, 2, data, nbytes, &NoOpDeallocator, NULL);
    return tensor;
}

// exec inférence

TF_Tensor* run_inference(TF_Session *session, TF_Graph *graph, TF_Tensor *input,
                         const char *input_op_name, const char *output_op_name) {
    TF_Status *status = TF_NewStatus();

    // identifie les opérations d'E/S dans le graphe
    TF_Output input_op = { TF_GraphOperationByName(graph, input_op_name), 0 };
    TF_Output output_op = { TF_GraphOperationByName(graph, output_op_name), 0 };

    if (input_op.oper == NULL || output_op.oper == NULL) {
        printf("ERROR: Failed to find input/output op\n");
        exit(1);
    }

    TF_Tensor *output_tensor = NULL;

    // exécution à proprement parler
    TF_SessionRun(session,
        NULL,
        &input_op, &input, 1,
        &output_op, &output_tensor, 1,
        NULL, 0, NULL,
        status);

    if (TF_GetCode(status) != TF_OK) {
        printf("ERROR: Failed to run inference: %s\n", TF_Message(status));
        exit(1);
    }

    TF_DeleteStatus(status);
    return output_tensor;
}

// affichage du contenu du tensor float

void print_tensor_float(TF_Tensor *tensor) {
    float *data = (float*)TF_TensorData(tensor);
    int count = TF_TensorByteSize(tensor) / sizeof(float);
    for (int i = 0; i < count; i++) {
        printf("  [%.3f]", data[i]);
    }
    printf("\n");
}
