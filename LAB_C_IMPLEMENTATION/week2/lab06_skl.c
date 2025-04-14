#include <stdio.h>

// dot product predict
float predict_house_price(const float *x, const float *w, float b, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * w[i];
    }
    return sum + b;
}

int main() {
    // 1200 sqft, 3 chambres, 1 étage, 40 ans
    float x_house[] = {1200, 3, 1, 40};
    int n = sizeof(x_house) / sizeof(x_house[0]);

    // coeff de sk learn
    float w[] = {0.257, 100, 50000, -150};
    float b = 50000;

    float predicted_price = predict_house_price(x_house, w, b, n);

    printf("Prédiction du prix d'une maison : %.2f dollars\n", predicted_price);

    return 0;
}
