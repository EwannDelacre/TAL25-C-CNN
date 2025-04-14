#include <stdio.h>
#include <math.h>

// fct sigmoide
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

int main() {
    printf("=== Fonction sigmoïde ===\n");
    printf("  z\t| sigmoid(z)\n");
    printf("-------------------------\n");

    // Plage de valeurs de -10 à 10 (inclus)
    for (int z = -10; z <= 10; z++) {
        float s = sigmoid((float)z);
        printf("%4d\t| %.6f\n", z, s);
    }

    return 0;
}
