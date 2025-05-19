//#include <zephyr/kernel.h>
#include <stdio.h>
#include "infer.h"

void main(void) {
    // data exemple
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[3] = {0};

    // init du modèle
    setup_model();

    // lancement inférence
    run_inference(input, 4, output, 3);

    printf("Résultat : %.3f %.3f %.3f\n", output[0], output[1], output[2]);
}
