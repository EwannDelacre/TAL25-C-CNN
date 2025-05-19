#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// params
#define IMG_HEIGHT 4
#define IMG_WIDTH 4
#define KERNEL_SIZE 2
#define N_CLASSES 3
#define CONV_STRIDE 1

// stockage features
typedef struct {
    int height;
    int width;
    float** data;
} FeatureMap;

// allocation features
FeatureMap* createFeatureMap(int height, int width) {
    FeatureMap* map = (FeatureMap*)malloc(sizeof(FeatureMap));
    map->height = height;
    map->width = width;
    
    map->data = (float**)malloc(height * sizeof(float*));
    for (int i = 0; i < height; i++) {
        map->data[i] = (float*)calloc(width, sizeof(float));
    }
    
    return map;
}

// free features
void freeFeatureMap(FeatureMap* map) {
    for (int i = 0; i < map->height; i++) {
        free(map->data[i]);
    }
    free(map->data);
    free(map);
}

// print features
void printFeatureMap(FeatureMap* map, const char* name) {
    printf("%s (%dx%d):\n", name, map->height, map->width);
    for (int i = 0; i < map->height; i++) {
        for (int j = 0; j < map->width; j++) {
            printf("%.4f ", map->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// convolution sur image 2D
FeatureMap* conv2d(FeatureMap* input, float kernel[KERNEL_SIZE][KERNEL_SIZE], float bias, int stride) {
    int output_height = (input->height - KERNEL_SIZE) / stride + 1;
    int output_width = (input->width - KERNEL_SIZE) / stride + 1;
    
    FeatureMap* output = createFeatureMap(output_height, output_width);
    
    // parcours image
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            float sum = 0.0f;
            
            // calcul
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    int input_y = y * stride + ky;
                    int input_x = x * stride + kx;
                    sum += input->data[input_y][input_x] * kernel[ky][kx];
                }
            }
            
            output->data[y][x] = sum + bias;
        }
    }
    
    return output;
}

// fct d'activation ReLU
void applyRelu(FeatureMap* map) {
    for (int i = 0; i < map->height; i++) {
        for (int j = 0; j < map->width; j++) {
            map->data[i][j] = fmaxf(0.0f, map->data[i][j]);
        }
    }
}

// GAP
float globalAveragePooling(FeatureMap* map) {
    float sum = 0.0f;
    int total = map->height * map->width;
    
    for (int i = 0; i < map->height; i++) {
        for (int j = 0; j < map->width; j++) {
            sum += map->data[i][j];
        }
    }
    
    return sum / total;
}

// softmax
void softmax(float input[], float output[], int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val); 
        sum += output[i];
    }
    
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// indice val max
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
    // init input
    FeatureMap* input = createFeatureMap(IMG_HEIGHT, IMG_WIDTH);
    float input_data[IMG_HEIGHT][IMG_WIDTH] = {
        {1, 2, 0, 1},
        {3, 1, 2, 1},
        {0, 1, 3, 2},
        {1, 0, 2, 3}
    };
    
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            input->data[i][j] = input_data[i][j];
        }
    }
    
    // filtres kernel
    float kernels[N_CLASSES][KERNEL_SIZE][KERNEL_SIZE] = {
        {{1, -1}, {0, 1}},   // Classe 0
        {{0, 1}, {1, -1}},   // Classe 1
        {{-1, 0}, {1, 1}}    // Classe 2
    };
    
    float biases[N_CLASSES] = {0.5, -0.3, 0.1};
    
    // results
    float pooling_out[N_CLASSES];
    float softmax_out[N_CLASSES];
    
    // print input
    printf("=== CNN Amélioré ===\n\n");
    printFeatureMap(input, "Image d'entrée");
    
    FeatureMap* conv_outputs[N_CLASSES];
    
    for (int c = 0; c < N_CLASSES; c++) {
        
        // convolution
        conv_outputs[c] = conv2d(input, kernels[c], biases[c], CONV_STRIDE);
        
        printf("Kernel %d:\n", c);
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                printf("%.2f ", kernels[c][i][j]);
            }
            printf("\n");
        }
        printf("Bias: %.2f\n\n", biases[c]);
        
        // appel ReLU
        applyRelu(conv_outputs[c]);
        
        // print features
        char name[50];
        sprintf(name, "Feature map %d (après ReLU)", c);
        printFeatureMap(conv_outputs[c], name);
        
        // appel GAP
        pooling_out[c] = globalAveragePooling(conv_outputs[c]);
        printf("Classe %d : Global Average Pooling = %.4f\n\n", c, pooling_out[c]);
    }
    
    // appele softmax
    softmax(pooling_out, softmax_out, N_CLASSES);
    
    // print probas
    printf("Probabilités:\n");
    for (int i = 0; i < N_CLASSES; i++) {
        printf("Classe %d : %.4f (%.2f%%)\n", i, softmax_out[i], softmax_out[i] * 100);
    }
    
    // prédiction de classe
    int predicted_class = argmax(softmax_out, N_CLASSES);
    printf("\n=> Classe prédite : %d\n", predicted_class);
    
    // free final
    freeFeatureMap(input);
    for (int c = 0; c < N_CLASSES; c++) {
        freeFeatureMap(conv_outputs[c]);
    }
    
    return 0;
}