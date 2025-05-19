#include <stdio.h>

// fct calcul produit scalaire entre 2 vecteurs
float dot_product(const float *a, const float *b, int n) {
    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// prédiction simple pour une ligne entrée
float predict(const float *x, const float *w, float b, int n) {
    return dot_product(x, w, n) + b;
}

// prédiction sur le tout
void predict_all(const float X[][4], const float *w, float b, int m, int n, float *y_pred) {
    for (int i = 0; i < m; i++) {
        y_pred[i] = predict(X[i], w, b, n);
    }
}

int main() {
    // input 3 exemples 4 features
    float X[3][4] = {
        {2104, 5, 1, 45},
        {1416, 3, 2, 40},
        {852,  2, 1, 35}
    };

    // valeur réelle de y
    float y_actual[3] = {460, 232, 178};

    // params
    float w[4] = {0.39133535, 18.75376741, -53.36032453, -26.42131618};
    float b = 785.1811367994083;

    float stock_pred[3];

    predict_all(X, w, b, 3, 4, stock_pred);

    // print
    printf("=== Prédictions multivariées ===\n");
    for (int i = 0; i < 3; i++) {
        printf("Exemple %d : prédiction = %.2f | valeur réelle = %.2f\n", i + 1, stock_pred[i], y_actual[i]);
    }

    return 0;
}
