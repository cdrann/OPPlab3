#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define answerExpected(i, j) answerExpected[K * i + j]

void printExpected(double *result, int sizeN, int sizeM, int sizeK) {
    puts("EXPECTED:");
    for (int i = 0; i < sizeM; i++) {
        for (int j = 0; j < sizeK; j++) {
            printf(" %3.1f", result[sizeK * i + j]);
        }
        printf("\n");
    }
    puts("");
}

void getExpected(double *result, double *matrix1, double *matrix2, int sizeN, int sizeM, int sizeK) {
    for (int i = 0; i < sizeN; ++i) {
        for (int j = 0; j < sizeK; ++j) {
            for (int k = 0; k < sizeM; ++k) {
                result[i * sizeK + j] += matrix1[i * sizeM + k] * matrix2[sizeK * k + j];
            }
        }
    }
    //printExpected(result, sizeN, sizeM, sizeK);
}

bool areEqual(double *C, double *expectedC, int sizeN, int sizeM, int sizeK) {
    for (int i = 0; i < sizeN; ++i) {
        for (int j = 0; j < sizeK; ++j) {
            for (int k = 0; k < sizeM; ++k) {
                if(C[i * sizeK + j] != expectedC[i * sizeK + j]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void printRes(double *C, int MyP, int sizeN, int sizeM, int sizeK) {
    if (MyP == 0) {
        puts("RESULT:");

        for (int i = 0; i < sizeM; i++) {
            for (int j = 0; j < sizeK; j++) {
                printf(" %3.1f", C[sizeK * i + j]);
            }
            printf("\n");
        }
    }
}


