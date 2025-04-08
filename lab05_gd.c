#include <stdio.h>

// fct cost : calcul de l'erreur quadra moyenne
float compute_cost(float x[], float y[], int m, float w, float b) {
    float cost_sum = 0.0;
    for (int i = 0; i < m; i++) {
        float f_wb = w * x[i] + b;
        float cost = (f_wb - y[i]) * (f_wb - y[i]);  // (prediction - actuelle)^2
        cost_sum += cost;
        // print
        //printf("x=%.2f, y=%.2f => f=%.2f, (f-y)^2=%.2f\n", x[i], y[i], f_wb, cost);
    }
    float total_cost = cost_sum / (2 * m);
    return total_cost;
}

// fct calcul gradient de w et b
void compute_gradient(float x[], float y[], int m, float w, float b, float* dj_dw, float* dj_db) {
    float sum_dw = 0.0;
    float sum_db = 0.0;

    for (int i = 0; i < m; i++) {
        float f_wb = w * x[i] + b;
        float error = f_wb - y[i];
        sum_dw += error * x[i];
        sum_db += error;
    }

    *dj_dw = sum_dw / m;
    *dj_db = sum_db / m;
}


// gradient descent
void gradient_descent(float x[], float y[], int m, float* w, float* b,
                      float alpha, int num_iters) {
    float cost, dj_dw, dj_db;

    for (int i = 0; i < num_iters; i++) {
        compute_gradient(x, y, m, *w, *b, &dj_dw, &dj_db);

        // màj params
        *w = *w - alpha * dj_dw;
        *b = *b - alpha * dj_db;

        // print
        cost = compute_cost(x, y, m, *w, *b);
        if (i % (num_iters / 10) == 0 || i == num_iters - 1) {
            printf("Itération %4d | Coût = %08.2f | grad selon w = %.4f | grad selon b = %.4f | w = %.4f | b = %.4f\n",
                   i, cost, dj_dw, dj_db, *w, *b);
        }
    }
}

// predi selon entrée x et sortie y
float predict(float x_input, float w, float b) {
    return w * x_input + b;
}

int main() {
    float x_train[] = {1.0, 2.0};
    float y_train[] = {300.0, 500.0};
    int m = sizeof(x_train) / sizeof(x_train[0]);

    // params
    float w = 0.0;
    float b = 0.0;
    float alpha = 0.01;
    int iterations = 10000;

    gradient_descent(x_train, y_train, m, &w, &b, alpha, iterations);

    printf("\nParamètres finaux : w = %.4f | b = %.4f\n", w, b);

    // pred avec fct de pred
    float pred1 = predict(1.0, w, b);
    float pred2 = predict(1.2, w, b);
    float pred3 = predict(2.0, w, b);

    printf("Prédiction pour 1000 sqft : %.1f mille dollars\n", pred1);
    printf("Prédiction pour 1200 sqft : %.1f mille dollars\n", pred2);
    printf("Prédiction pour 2000 sqft : %.1f mille dollars\n", pred3);

    return 0;
}
