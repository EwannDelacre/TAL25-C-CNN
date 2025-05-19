
#include "infer.h"
#include "model_data.h"  // tableau modèle

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// taille de l'arène mémoire des tenseurs
#define TENSOR_ARENA_SIZE 4096
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// objets TensorFlow
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

void setup_model() {
    // initialisation plateforme (souvent vide en POSIX)
    tflite::InitializeTarget();

    // conversion du tableau modèle
    const tflite::Model* model = tflite::GetModel(g_model);

    // verif compatibilité
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Erreur : version du modèle incompatible.\n");
        return;
    }

    // enregistrement opérations
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();

    // création interpréteur
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // allocation mémoire et recup tenseurs d'E/S
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Erreur lors de l'allocation des tenseurs.\n");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

// lancement inférence
// input > modèle > output
void run_inference(const float *input_data, int input_size, float *output_data, int output_size) {

  // input
    for (int i = 0; i < input_size; ++i) {
        input->data.f[i] = input_data[i];
    }

    // invocation modèle d'inférence
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Erreur : échec de l'inférence.\n");
        return;
    }

    // copie output
    for (int i = 0; i < output_size; ++i) {
        output_data[i] = output->data.f[i];
    }
}
