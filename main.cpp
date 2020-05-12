#include<cstdio>
#include<cstdlib>
#include<mpi.h>

#include "Test.h"

#define NUM_DIMS 2
#define M 16
#define N 16
#define K 16

void initABC(double *A, double *B, double *C, double *expected);


int MatrixMultiply_2D(double *A, double *B, double *C, int *p, MPI_Comm comm) {
    double *partA, *partB, *partC;
    int rank;
    int remains[2];
    int sizeofdouble, array_of_displacements[2];
    int *countc = nullptr, *dispc = nullptr, *array_of_blocklength_B = nullptr, *array_of_displacements_B = nullptr; //смещения и размер подматриц для сборки
    MPI_Datatype newtype = MPI_DATATYPE_NULL, newTypeC = MPI_DATATYPE_NULL, array_of_types[2];

    MPI_Comm comm_2D, comm_1D[NUM_DIMS], comm_copy;
    MPI_Comm_dup(comm, &comm_copy);
    MPI_Bcast(p, 2, MPI_INT, 0, comm_copy);
    int periods[NUM_DIMS] = {0};
    MPI_Cart_create(comm_copy, 2, p, periods, 0, &comm_2D); //2d решетка P0 x P1

    MPI_Comm_rank(comm_2D, &rank);
    int coords[NUM_DIMS];
    MPI_Cart_coords(comm_2D, rank, NUM_DIMS, coords);

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            remains[j] = (i == j);
        }
        MPI_Cart_sub(comm_2D, remains, &comm_1D[i]);
    }

    int localM = M / p[0];
    int localK = K / p[1];

    partA = (double *) malloc(localM * N * sizeof(double));
    partB = (double *) malloc(N * localK * sizeof(double));
    partC = (double *) malloc(localM * localK * sizeof(double));
    int array_of_blocklengths[2] = {1};

    if(rank == 0) {
        //Устанавливаем типы для полос B и подматриц С
        MPI_Type_vector(N, localK, K, MPI_DOUBLE, &array_of_types[0]);
        //корректируем диапазион размера полосы
        MPI_Aint lb;
        MPI_Type_get_extent(MPI_DOUBLE, &lb, &sizeofdouble);

        array_of_displacements[0] = 0;
        array_of_displacements[1] = sizeofdouble * localK;
        array_of_types[1] = MPI_UB;
        MPI_Type_create_struct(NUM_DIMS, array_of_blocklengths, array_of_displacements, array_of_types, &newtype);
        MPI_Type_commit(&newtype);

        array_of_displacements_B = (int *) malloc(p[1] * sizeof(int));
        array_of_blocklength_B = (int *) malloc(p[1] * sizeof(int));
        for(int j = 0; j < p[1]; j++) {
            array_of_displacements_B[j] = j;
            array_of_blocklength_B[j] = 1;
        }

        MPI_Type_vector(localM, localK, K, MPI_DOUBLE, &array_of_types[0]);
        MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &newTypeC);
        MPI_Type_commit(&newTypeC);

        dispc = (int *) malloc(p[0] * p[1] * sizeof(int));
        countc = (int *) malloc(p[0] * p[1] * sizeof(int));
        for(int i = 0; i < p[0]; i++) {
            for(int j = 0; j < p[1]; j++) {
                dispc[i * p[1] + j]= (i * p[1] * localM + j);
                countc[i * p[1] + j] = 1;
            }
        }
    }

    // 1. Матрица А распределяется по горизонтальным полосам вдоль координаты (x,0). [0я ветвь - scatter горизонтальные A по x]
    if(coords[1] == 0) {
        MPI_Scatter(A, localM * N, MPI_DOUBLE, partA,localM * N, MPI_DOUBLE, 0, comm_1D[0]);
    }

    // 2. Матрица B распределяется по вертикальным полосам вдоль координаты (0,y). [0я ветвь - scatter горизонтальные B по y]
    if(coords[0] == 0) {
        MPI_Scatterv(B, array_of_blocklength_B, array_of_displacements_B, newtype, partB, N * localK, MPI_DOUBLE, 0, comm_1D[1]);
    }

    // 3. Полосы А(partA) распространяются в измерении y.
    MPI_Bcast(partA, localM * N, MPI_DOUBLE, 0, comm_1D[1]);

    // 4. Полосы B(partB) распространяются в измерении х.
    MPI_Bcast(partB, N * localK, MPI_DOUBLE, 0, comm_1D[0]);

    // 5. Каждый процесс вычисляет одну подматрицу произведения partC
    for(int i = 0; i < localM; i++) {
        for(int j = 0; j < localK; j++) {
            partC[localK * i + j] = 0.0;
            for(int k = 0; k < N; k++) {
                partC[localK * i + j] = partC[localK * i + j] + partA[N * i + j] * partB[localK * i + j];
            }
        }
    }

    // 6. Матрица C собирается из (x,y) плоскости, в ветви 0 собираем все подматрицы PartC
    MPI_Gatherv(partC, localM * localK, MPI_DOUBLE, C, countc, dispc, newTypeC, 0, comm_2D);

    free(partA);
    free(partB);
    free(partC);
    MPI_Comm_free(&comm_copy);
    MPI_Comm_free(&comm_2D);
    for(int & i : comm_1D) {
        MPI_Comm_free(&i);
    }
    if(rank == 0) {
        free(countc);
        free(dispc);
        MPI_Type_free(&newtype);
        MPI_Type_free(&newTypeC);
        MPI_Type_free(&array_of_types[0]);
    }
    return 0;
}

int main(int argc, char **argv) {
    int err = 0;
    int mnk[3], p[2]; // p -- кол-во процессов по каждой координате
    int dims[NUM_DIMS] = {0};
    double *A = nullptr, *B = nullptr, *C = nullptr, *expected = nullptr;

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc == 3) {
        p[0] = strtol(argv[1], nullptr, 10);
        p[1] = strtol(argv[2], nullptr, 10);
        if(p[0] * p[1] != size) {
            printf("Error. Wrong size of lattice\n");
            MPI_Abort(MPI_COMM_WORLD, err);
            return 1;
        }
    } else {
        p[0] = 1;
        p[1] = size;
    }

    MPI_Dims_create(size, NUM_DIMS, dims);
    MPI_Comm comm;
    int periods[NUM_DIMS] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm); // reorder - нумерация сохранена

    if (rank == 0) {
        mnk[0] = M;
        mnk[1] = N;
        mnk[2] = K;

        A = (double *) malloc(M * N * sizeof(double));
        B = (double *) malloc(N * K * sizeof(double));
        C = (double *) malloc(M * K * sizeof(double));
        expected = (double *) malloc(M * K * sizeof(double));

        initABC(A, B, C, expected);

        getExpected(expected, A, B, N, M, K);
    }

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    MatrixMultiply_2D(A, B, C, p, comm);
    //printRes(C, rank, N, M, K);

    if(rank == 0) {
        end_time = MPI_Wtime();
        printf("Processes: %d; Time: %f\n", size, end_time - start_time);
    }

    if (rank == 0) {
        if(areEqual(C, expected, N, M, K)) {
            puts("Verdict: Accepted.");
        } else {
            puts("Verdict: WA. Try again.");
        }

        free(A);
        free(B);
        free(C);
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return (0);
}

void initABC(double *A, double *B, double *C, double *expected) {
    int err = 0;
    if(A == nullptr || B == nullptr || C == nullptr || expected == nullptr) {
        printf("Allocation fail!\n");
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[N * i + j] = 1;
        }
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B[K * j + k] = k + 1;
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