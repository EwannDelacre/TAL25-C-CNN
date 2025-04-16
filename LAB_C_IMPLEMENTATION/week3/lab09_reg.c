#include <stdio.h>
#include <math.h>

// fct sigmoide
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// prédiction logistique
float logistic_predict(const float *x, const float *w, float b, int n) {
    float z = 0.0f;
    for (int i = 0; i < n; i++) {
        z += x[i] * w[i];
    }
    z += b;
    return sigmoid(z);
}

int predict_class(float x[], float w[], float b, int n) {
    float prob = logistic_predict(x, w, b, n);
    return prob >= 0.5f ? 1 : 0;
}

int main() {
    // params
    float w[] = {0.5, -0.3, 0.8};
    float b = -0.1;

    // data
    float X[][3] = {
        {0.2, 0.7, 1.5},
        {1.0, -0.5, 0.3},
        {0.8, 0.1, 0.6},
        {1.2, -0.8, 0.5}
    };
    int y_true[] = {1, 0, 1, 0};

    int m = sizeof(X) / sizeof(X[0]);
    int n = sizeof(X[0]) / sizeof(X[0][0]);

    printf("=== Prédictions logistiques avec sigmoïde ===\n");
    for (int i = 0; i < m; i++) {
        float z = 0.0;
        for (int j = 0; j < n; j++) z += X[i][j] * w[j];
        z += b;
        float prob = sigmoid(z);
        int pred = prob >= 0.5 ? 1 : 0;

        printf("Exemple %d: z = %.3f | sigmoid(z) = %.3f | prédiction = %d | réel = %d\n",
               i + 1, z, prob, pred, y_true[i]);
    }

    return 0;
}
