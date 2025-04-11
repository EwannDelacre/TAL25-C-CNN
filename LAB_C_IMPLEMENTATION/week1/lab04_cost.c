#include <stdio.h>

// fct cost : calcul de l'erreur quadra moyenne
float compute_cost(float x[], float y[], int m, float w, float b) {
    float cost_sum = 0.0;
    for (int i = 0; i < m; i++) {
        float f_wb = w * x[i] + b;
        float cost = (f_wb - y[i]) * (f_wb - y[i]);  // (prediction - actuelle)^2
        cost_sum += cost;
        // print
        printf("x=%.2f, y=%.2f => f=%.2f, (f-y)^2=%.2f\n", x[i], y[i], f_wb, cost);
    }
    float total_cost = cost_sum / (2 * m);
    return total_cost;
}

int main() {
    // input
    float x_train[] = {1.0, 2.0};
    float y_train[] = {300.0, 500.0};
    int m = sizeof(x_train) / sizeof(x_train[0]);

    // param
    float w = 200.0;
    float b = 100.0;

    // calcul (appel fct) et print
    float cost = compute_cost(x_train, y_train, m, w, b);
    printf("\n>> Coût total : %.2f\n", cost);

    return 0;
}
