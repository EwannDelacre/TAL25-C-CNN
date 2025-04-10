#include <stdio.h>

// fct calcul produit scalaire entre 2 vecteurs
float dot_product(const float *a, const float *b, int n) {
    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// prédiction vectorisée
void predict(const float **X, const float *w, float b, int m, int n, float *y_pred) {
    for (int i = 0; i < m; i++) {
        y_pred[i] = dot_product(X[i], w, n) + b;
    }
}

// affichage vecteurs
void print_vector(const float *v, int n, const char *label) {
    printf("%s = [", label);
    for (int i = 0; i < n; i++) {
        printf("%.2f", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    // input
    float x0[] = {1.0, 2.0, 3.0};
    float x1[] = {4.0, 5.0, 6.0};
    float x2[] = {7.0, 8.0, 9.0};
    float x3[] = {1.5, 2.5, 3.5};
    const float *X[] = {x0, x1, x2, x3};

    int m = 4; // nb exemples
    int n = 3; // nb features

    // params
    float w[] = {0.1, 0.2, 0.3};
    float b = 1.0;

    float stock_pred[m];

    // Prédiction
    predict(X, w, b, m, n, stock_pred);

    // Affichage
    printf("=== Prédictions vectorisées ===\n");
    for (int i = 0; i < m; i++) {
        printf("x[%d] -> prédiction = %.2f\n", i, stock_pred[i]);
    }

    // Affichage des paramètres
    print_vector(w, n, "\nPoids w");
    printf("Biais b = %.2f\n", b);

    return 0;
}
