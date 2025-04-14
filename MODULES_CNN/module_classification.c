#include <stdio.h>
#include <math.h>

// fct sigmoide
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// produit scalaire et biais
float linear_combination(const float *x, const float *w, float b, int n) {
    float z = 0.0f;
    for (int i = 0; i < n; i++) {
        z += x[i] * w[i];
    }
    return z + b;
}

// prediction lineaire classfication avec la fct sigmoide
int predict_class(const float *x, const float *w, float b, int n) {
    float z = linear_combination(x, w, b, n);
    float y_prob = sigmoid(z);
    return y_prob >= 0.5f ? 1 : 0;
}

int main() {
    // params
    float w[] = {1.2, -0.8, 0.5}; // 3 features
    float b = -0.3;

    // data
    float X[][3] = {
        {2.0, 1.0, 0.5},
        {0.5, 2.0, 1.5},
        {1.0, 0.5, 3.0},
        {2.0, 2.0, 2.0}
    };
    int y_true[] = {1, 0, 1, 1};
    int m = sizeof(X) / sizeof(X[0]);
    int n = sizeof(X[0]) / sizeof(X[0][0]);

    printf("=== Prédiction binaire avec sigmoïde ===\n");
    for (int i = 0; i < m; i++) {
        float z = linear_combination(X[i], w, b, n);
        float prob = sigmoid(z);
        int pred = prob >= 0.5 ? 1 : 0;
        printf("Exemple %d: z = %.3f | sigmoid(z) = %.3f | prédiction = %d | attendu = %d\n",
               i + 1, z, prob, pred, y_true[i]);
    }

    return 0;
}
