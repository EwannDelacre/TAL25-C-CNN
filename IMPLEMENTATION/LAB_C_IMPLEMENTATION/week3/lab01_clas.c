#include <stdio.h>

// prediction linéaire binaire
int predict_class(float x, float w, float b) {
    float z = w * x + b;
    return z >= 0 ? 1 : 0;
}

int main() {
    // data
    float x_train[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    int y_train[]    = {  0,   0,   0,   1,   1,   1};
    int m = sizeof(x_train) / sizeof(x_train[0]);

    // params
    float w = 1.0;
    float b = -2.5;

    printf("=== Classification binaire (1 variable) ===\n");
    for (int i = 0; i < m; i++) {
        float x = x_train[i];
        int prediction = predict_class(x, w, b);
        printf("x = %.1f | prédiction = %d | valeur réelle = %d\n", x, prediction, y_train[i]);
    }

    return 0;
}
