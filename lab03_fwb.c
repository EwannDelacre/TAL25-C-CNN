#include <stdio.h>

// fct prédiction linéaire simple
float predict(float x, float w, float b) {
    return w * x + b;
}

int main() {
    // params
    float w = 200.0;
    float b = 100.0;

    // entrées (à modifier pour un flot ou un fichier)
    float x_inputs[] = {1.0, 1.2, 2.0, 2.5};
    int n = sizeof(x_inputs) / sizeof(x_inputs[0]);

    printf("=== Phase de prédiction ===\n");
    for (int i = 0; i < n; i++) {
        float y_pred = predict(x_inputs[i], w, b);
        printf("x = %.1f >>> prédiction = %.1f milliers de dollars\n", x_inputs[i], y_pred);
    }

    return 0;
}
