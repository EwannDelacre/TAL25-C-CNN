#include <stdio.h>
#include <math.h>
#include <float.h>

#define IMG_SIZE 4
#define KERNEL_SIZE 2
#define N_CLASSES 3

// convolution sur une zone 2x2
float conv2d_2x2(float input[IMG_SIZE][IMG_SIZE], float kernel[KERNEL_SIZE][KERNEL_SIZE], float bias) {
    float sum = 0.0f;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            sum += input[i][j] * kernel[i][j];
        }
    }
    return sum + bias;
}

// fcrt ReLU
float relu(float x) {
    return fmaxf(0.0f, x);
}

// fct Softmax
void softmax(float input[], float output[], int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (input[i] > max_val) max_val = input[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val); // for numerical stability
        sum += output[i];
    }
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

int argmax(float *arr, int n) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    // data (image 4x4)
    float input[IMG_SIZE][IMG_SIZE] = {
        {1, 2, 0, 1},
        {3, 1, 2, 1},
        {0, 1, 3, 2},
        {1, 0, 2, 3}
    };

    // simule 3 filtres de conv pour 3 classes
    float kernels[N_CLASSES][KERNEL_SIZE][KERNEL_SIZE] = {
        {{1, -1}, {0, 1}},  // classe 0
        {{0, 1}, {1, -1}},  // classe 1
        {{-1, 0}, {1, 1}}   // classe 2
    };
    float biases[N_CLASSES] = {0.5, -0.3, 0.1};

    // résultat
    float conv_out[N_CLASSES];
    float softmax_out[N_CLASSES];

    printf("=== CNN Delacre (v1) ===\n");
    for (int c = 0; c < N_CLASSES; c++) {
        float z = conv2d_2x2(input, kernels[c], biases[c]);
        conv_out[c] = relu(z);
        printf("Classe %d : conv+relu = %.4f\n", c, conv_out[c]);
    }

    softmax(conv_out, softmax_out, N_CLASSES);
    printf("\n Probabilités:\n");
    for (int i = 0; i < N_CLASSES; i++) {
        printf("Classe %d : %.4f\n", i, softmax_out[i]);
    }

    int predicted_class = argmax(softmax_out, N_CLASSES);
    printf("\n=> Classe prédite : %d\n", predicted_class);

    return 0;
}
