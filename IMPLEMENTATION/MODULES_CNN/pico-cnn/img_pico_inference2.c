/*

Remarques pour la compilation:

Ce code nécessite l'installation de pico-cnn ainsi que les libraires STB.
Assurez vous de compiler avec les bons flags :

gcc img_pico_inference2.c -o img_pico_inference2 -I./pico-cnn/include -L./pico-cnn/lib -lpico-cnn -lm

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

// Inclusions pour pico-cnn (chemins corrects)
#include "pico-cnn/pico-cnn/pico-cnn.h"
#include "pico-cnn/pico-cnn/layers/convolution.h"
#include "pico-cnn/pico-cnn/layers/fully_connected.h"
#include "pico-cnn/pico-cnn/layers/pooling/max_pooling.h"
#include "pico-cnn/pico-cnn/layers/activation_functions/relu.h"
#include "pico-cnn/pico-cnn/io/read_binary_weights.h"

// Inclusion pour le traitement d'images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

// Dimensions MNIST
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define NUM_CLASSES 10

// Structure pour notre modèle CNN
typedef struct {
    struct convolution_layer *conv1;
    struct max_pooling_layer *pool1;
    struct convolution_layer *conv2;
    struct max_pooling_layer *pool2;
    struct fully_connected_layer *fc;
} mnist_model_t;

// Fonction pour vérifier les allocations mémoire
void check_allocation(void *ptr, const char *name) {
    if (ptr == NULL) {
        fprintf(stderr, "Erreur: Échec de l'allocation mémoire pour %s\n", name);
        exit(1);
    }
}

// Libération des ressources du modèle
void free_mnist_model(mnist_model_t *model) {
    if (model) {
        if (model->conv1) free_convolution_layer(model->conv1);
        if (model->pool1) free_max_pooling_layer(model->pool1);
        if (model->conv2) free_convolution_layer(model->conv2);
        if (model->pool2) free_max_pooling_layer(model->pool2);
        if (model->fc) free_fully_connected_layer(model->fc);
        free(model);
    }
}

// Création du modèle CNN pour MNIST
mnist_model_t* create_mnist_model() {
    mnist_model_t *model = (mnist_model_t*)malloc(sizeof(mnist_model_t));
    check_allocation(model, "modèle MNIST");

    // Configuration des couches
    // Première couche de convolution: 16 filtres de 3x3, avec padding same
    model->conv1 = create_convolution_layer(1, 28, 28, 16, 3, 3, 1, 1, SAME);
    check_allocation(model->conv1, "conv1");

    // Première couche de pooling: 2x2 avec stride 2
    model->pool1 = create_max_pooling_layer(16, 28, 28, 2, 2, 2, 2);
    check_allocation(model->pool1, "pool1");

    // Deuxième couche de convolution: 32 filtres de 3x3, avec padding same
    model->conv2 = create_convolution_layer(16, 14, 14, 32, 3, 3, 1, 1, SAME);
    check_allocation(model->conv2, "conv2");

    // Deuxième couche de pooling: 2x2 avec stride 2
    model->pool2 = create_max_pooling_layer(32, 14, 14, 2, 2, 2, 2);
    check_allocation(model->pool2, "pool2");

    // Couche fully connected: entrée 7x7x32 = 1568, sortie 10 (classes)
    model->fc = create_fully_connected_layer(7 * 7 * 32, NUM_CLASSES);
    check_allocation(model->fc, "fc");

    return model;
}

// Initialisation des poids du modèle (pour simulation)
void initialize_model_weights(mnist_model_t *model) {
    // Initialisation des poids pour conv1
    for (int i = 0; i < model->conv1->kernel_size; i++) {
        model->conv1->weights[i] = ((float)(i % 9 - 4)) / 10.0f;
    }
    for (int i = 0; i < model->conv1->n_filters; i++) {
        model->conv1->biases[i] = 0.1f;
    }

    // Initialisation des poids pour conv2
    for (int i = 0; i < model->conv2->kernel_size; i++) {
        model->conv2->weights[i] = ((float)(i % 9 - 4)) / 20.0f;
    }
    for (int i = 0; i < model->conv2->n_filters; i++) {
        model->conv2->biases[i] = 0.1f;
    }

    // Initialisation des poids pour la couche fully connected
    for (int i = 0; i < model->fc->weights_size; i++) {
        model->fc->weights[i] = ((float)(i % 10 - 5)) / (10.0f * (7 * 7 * 32));
    }
    for (int i = 0; i < model->fc->output_size; i++) {
        model->fc->biases[i] = 0.1f;
    }
}

// Chargement des poids depuis un fichier binaire
int load_model_weights(mnist_model_t *model, const char *weights_file) {
    FILE *fp = fopen(weights_file, "rb");
    if (!fp) {
        fprintf(stderr, "Erreur: Impossible d'ouvrir le fichier de poids %s\n", weights_file);
        return 0;
    }

    // Lecture des poids pour chaque couche
    // Conv1 weights
    if (fread(model->conv1->weights, sizeof(float), model->conv1->kernel_size, fp) != model->conv1->kernel_size) {
        fprintf(stderr, "Erreur: Échec de la lecture des poids conv1\n");
        fclose(fp);
        return 0;
    }

    // Conv1 biases
    if (fread(model->conv1->biases, sizeof(float), model->conv1->n_filters, fp) != model->conv1->n_filters) {
        fprintf(stderr, "Erreur: Échec de la lecture des biais conv1\n");
        fclose(fp);
        return 0;
    }

    // Conv2 weights
    if (fread(model->conv2->weights, sizeof(float), model->conv2->kernel_size, fp) != model->conv2->kernel_size) {
        fprintf(stderr, "Erreur: Échec de la lecture des poids conv2\n");
        fclose(fp);
        return 0;
    }

    // Conv2 biases
    if (fread(model->conv2->biases, sizeof(float), model->conv2->n_filters, fp) != model->conv2->n_filters) {
        fprintf(stderr, "Erreur: Échec de la lecture des biais conv2\n");
        fclose(fp);
        return 0;
    }

    // FC weights
    if (fread(model->fc->weights, sizeof(float), model->fc->weights_size, fp) != model->fc->weights_size) {
        fprintf(stderr, "Erreur: Échec de la lecture des poids fc\n");
        fclose(fp);
        return 0;
    }

    // FC biases
    if (fread(model->fc->biases, sizeof(float), model->fc->output_size, fp) != model->fc->output_size) {
        fprintf(stderr, "Erreur: Échec de la lecture des biais fc\n");
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

// Prétraitement de l'image
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

// Implémentation de softmax
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

// Fonction d'inférence principale
void infer_mnist_image(float* image_data, mnist_model_t* model) {
    // Allouer l'espace pour les résultats intermédiaires
    int output_size_conv1 = 28 * 28 * 16;  // Après conv1
    int output_size_pool1 = 14 * 14 * 16;  // Après pool1
    int output_size_conv2 = 14 * 14 * 32;  // Après conv2
    int output_size_pool2 = 7 * 7 * 32;    // Après pool2
    int output_size_fc = NUM_CLASSES;      // Sortie finale

    float *conv1_output = (float*)malloc(output_size_conv1 * sizeof(float));
    float *pool1_output = (float*)malloc(output_size_pool1 * sizeof(float));
    float *conv2_output = (float*)malloc(output_size_conv2 * sizeof(float));
    float *pool2_output = (float*)malloc(output_size_pool2 * sizeof(float));
    float *fc_output = (float*)malloc(output_size_fc * sizeof(float));

    check_allocation(conv1_output, "conv1_output");
    check_allocation(pool1_output, "pool1_output");
    check_allocation(conv2_output, "conv2_output");
    check_allocation(pool2_output, "pool2_output");
    check_allocation(fc_output, "fc_output");

    // Exécution des couches
    // Conv1 + ReLU
    convolution_forward(model->conv1, image_data, conv1_output);
    relu_forward(conv1_output, output_size_conv1);

    // Pool1
    max_pooling_forward(model->pool1, conv1_output, pool1_output);

    // Conv2 + ReLU
    convolution_forward(model->conv2, pool1_output, conv2_output);
    relu_forward(conv2_output, output_size_conv2);

    // Pool2
    max_pooling_forward(model->pool2, conv2_output, pool2_output);

    // Fully connected + Softmax
    fully_connected_forward(model->fc, pool2_output, fc_output);

    // Appliquer softmax pour obtenir les probabilités
    float probabilities[NUM_CLASSES];
    softmax(fc_output, probabilities, NUM_CLASSES);

    // Afficher les résultats
    int predicted_digit = 0;
    float max_prob = probabilities[0];

    printf("Probabilités par classe:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Chiffre %d: %.2f%%\n", i, probabilities[i] * 100);

        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            predicted_digit = i;
        }
    }

    printf("\n=> Chiffre prédit: %d (confiance: %.2f%%)\n", predicted_digit, max_prob * 100);

    // Libérer la mémoire
    free(conv1_output);
    free(pool1_output);
    free(conv2_output);
    free(pool2_output);
    free(fc_output);
}

// Mesure le temps d'inférence
double measure_inference_time(float* image_data, mnist_model_t* model, int num_iterations) {
    clock_t start, end;
    double total_time = 0.0;

    for (int i = 0; i < num_iterations; i++) {
        start = clock();

        // Allouer l'espace pour les résultats intermédiaires
        int output_size_conv1 = 28 * 28 * 16;
        int output_size_pool1 = 14 * 14 * 16;
        int output_size_conv2 = 14 * 14 * 32;
        int output_size_pool2 = 7 * 7 * 32;
        int output_size_fc = NUM_CLASSES;

        float *conv1_output = (float*)malloc(output_size_conv1 * sizeof(float));
        float *pool1_output = (float*)malloc(output_size_pool1 * sizeof(float));
        float *conv2_output = (float*)malloc(output_size_conv2 * sizeof(float));
        float *pool2_output = (float*)malloc(output_size_pool2 * sizeof(float));
        float *fc_output = (float*)malloc(output_size_fc * sizeof(float));

        // Exécution des couches
        convolution_forward(model->conv1, image_data, conv1_output);
        relu_forward(conv1_output, output_size_conv1);
        max_pooling_forward(model->pool1, conv1_output, pool1_output);
        convolution_forward(model->conv2, pool1_output, conv2_output);
        relu_forward(conv2_output, output_size_conv2);
        max_pooling_forward(model->pool2, conv2_output, pool2_output);
        fully_connected_forward(model->fc, pool2_output, fc_output);

        end = clock();
        total_time += ((double) (end - start)) / CLOCKS_PER_SEC;

        // Libérer la mémoire
        free(conv1_output);
        free(pool1_output);
        free(conv2_output);
        free(pool2_output);
        free(fc_output);
    }

    return total_time / num_iterations;
}

// Fonction principale
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    // Chargement de l'image
    const char* image_path = argv[1];
    int width, height, channels;
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 0);
    if (!image_data) {
        fprintf(stderr, "Impossible de charger l'image %s\n", image_path);
        return 1;
    }

    printf("Image chargée: %s (%dx%d avec %d canaux)\n", image_path, width, height, channels);

    // Prétraitement de l'image
    float preprocessed_data[MNIST_WIDTH * MNIST_HEIGHT];

    // Si les dimensions ne sont pas 28x28, redimensionner l'image
    if (width != MNIST_WIDTH || height != MNIST_HEIGHT) {
        printf("Redimensionnement de l'image %dx%d vers %dx%d\n", width, height, MNIST_WIDTH, MNIST_HEIGHT);

        // Créer un buffer temporaire pour l'image redimensionnée
        unsigned char* resized_data = (unsigned char*)malloc(MNIST_WIDTH * MNIST_HEIGHT * channels);
        check_allocation(resized_data, "resized_data");

        stbir_resize_uint8_linear(image_data, width, height, 0,
                           resized_data, MNIST_WIDTH, MNIST_HEIGHT, 0, channels);

        // Prétraiter l'image redimensionnée
        preprocess_image(resized_data, MNIST_WIDTH, MNIST_HEIGHT, channels, preprocessed_data);
        free(resized_data);
    } else {
        // Prétraiter l'image originale
        preprocess_image(image_data, width, height, channels, preprocessed_data);
    }

    printf("Échantillon de l'image prétraitée (10 premiers pixels):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", preprocessed_data[i]);
    }
    printf("...\n");

    // Affichage d'une représentation visuelle de l'image
    printf("Représentation visuelle de l'image:\n");
    for (int i = 0; i < MNIST_HEIGHT; i++) {
        for (int j = 0; j < MNIST_WIDTH; j++) {
            float pixel_val = preprocessed_data[i * MNIST_WIDTH + j];
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

    // Création du modèle CNN
    mnist_model_t* model = create_mnist_model();

    // Option 1: Charger les poids depuis un fichier (si disponible)
    if (access("weights.bin", F_OK) == 0) {
        printf("Chargement des poids depuis weights.bin...\n");
        if (!load_model_weights(model, "weights.bin")) {
            fprintf(stderr, "Erreur lors du chargement des poids. Utilisation des poids par défaut.\n");
            initialize_model_weights(model);
        }
    } else {
        // Option 2: Initialiser les poids pour la démonstration
        printf("Fichier de poids non trouvé. Initialisation des poids pour la démonstration...\n");
        initialize_model_weights(model);
    }

    // Exécuter l'inférence
    infer_mnist_image(preprocessed_data, model);

    // Mesurer le temps d'inférence (moyenne sur 10 itérations)
    double avg_time = measure_inference_time(preprocessed_data, model, 10);
    printf("\nTemps d'inférence moyen (sur 10 itérations): %.5f secondes\n", avg_time);

    // Libérer les ressources
    free_mnist_model(model);
    stbi_image_free(image_data);

    return 0;
}
