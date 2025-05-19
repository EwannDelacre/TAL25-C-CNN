#include <stdio.h>
#include <stdlib.h>
#include "tf_infer.h"

int main() {
    // data
    float input_data[4] = {1.0, 2.0, 3.0, 4.0};
    int input_size = 4;

    // charge le modèle .pb
    TF_Graph *graph = load_graph("../model/");  // dir
    TF_Session *session = create_session(graph);

    // création tensor d'entrée
    TF_Tensor *input_tensor = make_input_tensor(input_data, 1, input_size);

    const char *input_op_name = "serving_default_input:0";
    const char *output_op_name = "StatefulPartitionedCall:0";

    // exec du graphe
    TF_Tensor *output_tensor = run_inference(session, graph, input_tensor, input_op_name, output_op_name);

    // affichage
    float *output_data = (float*)TF_TensorData(output_tensor);
    int n_classes = TF_TensorByteSize(output_tensor) / sizeof(float);

    printf("\n=== Prédiction multiclasse ===\n");
    for (int i = 0; i < n_classes; i++) {
        printf("Classe %d : %.4f\n", i, output_data[i]);
    }

    // argmax
    int max_idx = 0;
    float max_val = output_data[0];
    for (int i = 1; i < n_classes; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = i;
        }
    }
    printf("=> Classe prédite : %d\n", max_idx);

    // clean
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_DeleteSession(session, NULL);
    TF_DeleteGraph(graph);

    return 0;
}
