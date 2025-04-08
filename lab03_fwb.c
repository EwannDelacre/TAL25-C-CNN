#include <stdio.h>

// fct regression linéaire
void compute_model_output(float x[], float f_wb[], int m, float w, float b) {
    for (int i = 0; i < m; i++) {
        f_wb[i] = w * x[i] + b;
    }
}

int main() {
    // input (taille en milliers de feet²)
    float x_train[] = {1.0, 2.0};
    float y_train[] = {300.0, 500.0};
    int m = sizeof(x_train) / sizeof(x_train[0]);

    // param
    float w = 200.0;
    float b = 500.0;

    // predictions
    float f_wb[m];

    // calcul predictions (appel fct quoi)
    compute_model_output(x_train, f_wb, m, w, b);

    // print
    printf("Prédictions du modèle linéaire : \n");
    for (int i = 0; i < m; i++) {
        printf("x = %.1f >>> prédiction = %.1f (valeur réelle = %.1f)\n", x_train[i], f_wb[i], y_train[i]);
    }

    // exemple pour 1.2
    float x_test = 1.2;
    float prediction = w * x_test + b;
    printf("\nPour x = %.1f >>> prédiction = %.0f milliers de dollars\n", x_test, prediction);

    return 0;
}
