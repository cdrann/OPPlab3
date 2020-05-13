//
// Created by Admin on 05.05.2020.
//


#include <cstdio>
#include <mpi.h>
#include <cstdlib>

void initABC(double *A, double *B, double *C, double *expected, int M, int N, int K) {
    int err = 0;
    if(A == nullptr || B == nullptr || C == nullptr || expected == nullptr) {
        printf("Allocation fail!\n");
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[N * i + j] = 3;
            if(i + 1 == j) {
                A[N * i + j] = 10;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B[K * j + k] = 0;
            if(k == j) {
                B[K * j + k] = 2;
            }
            if(k == 3 && j == 3) {
                B[K * j + k] = 30;
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            C[K * i + k] = 0.0;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            expected[K * i + k] = 0.0;
        }
    }
}