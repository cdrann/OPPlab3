#include<cstdio>
#include<cstdlib>
#include<mpi.h>

#include "Test.h"
#include "InitABC.h"
#define NUM_DIMS 2
#define M 16
#define N 16
#define K 16

void setType(int count, int blocklength, int *arrOfTypes, int *arrOfBlocklengths, int *arrOfDisplacements, MPI_Datatype *newType) {
    // Задание типа данных для вертикальной полосы в В / подматрицы partC в C
    MPI_Type_vector(count, blocklength, K, MPI_DOUBLE, &arrOfTypes[0]);
    // корректируем диапазон размера полосы/размер диапазона
    MPI_Type_create_struct(2, arrOfBlocklengths, arrOfDisplacements, arrOfTypes, newType);
    MPI_Type_commit(newType);
}

int MatrixMultiply_2D(double *A, double *B, double *C, int *p, MPI_Comm comm) {
    MPI_Comm comm_2D, comm_1D[2], comm_dup;

    // Во всех ветвях задаем подматрицы (полосы) [делим без остатка]
    // тут кратны.
    int localM = M / p[0];
    int localK = K / p[1];

    MPI_Aint lb, doubleExtent;
    MPI_Type_get_extent(MPI_DOUBLE, &lb, &doubleExtent);

    MPI_Comm_dup(comm, &comm_dup);

    MPI_Bcast(p, NUM_DIMS, MPI_INT, 0, comm_dup);

    // Создаем 2D решетку компьютеров размером p[0]*p[1]
    int periodic[2] = {0, 0};
    MPI_Cart_create(comm_dup, 2, p, periodic, 0, &comm_2D);

    // Находим порядковые номера ветвей и декартовы координаты ветвей в этой решетке
    int rank;
    int coords[2];
    MPI_Comm_rank(comm_2D, &rank);
    MPI_Cart_coords(comm_2D, rank, 2, coords);

    // Нахождение коммуникаторов для подрешеток 1D для рассылки полос матриц A и B
    int remains[2];
    for (int i = 0; i < NUM_DIMS; i++) {
        for (int j = 0; j < NUM_DIMS; j++) {
            remains[j] = (int) (i == j);
        }
        MPI_Cart_sub(comm_2D, remains, &comm_1D[i]);
    }

    auto *partA = (double *) malloc(localM * N * sizeof(double));
    auto *partB = (double *) malloc(N * localK * sizeof(double));
    auto *partC = (double *) malloc(localM * localK * sizeof(double));

    // Типы данных и массивы для создаваемых типов
    MPI_Datatype typeb = MPI_DATATYPE_NULL, typec = MPI_DATATYPE_NULL;

    if (rank == 0) {
        int arrOfBlocklengths[2] = {1, 1};
        int arrOfDisplacements[2] = {0, doubleExtent * localK};
        MPI_Datatype arrOfTypes[2];
        arrOfTypes[1] = MPI_UB;

        setType(N, localK, arrOfTypes, arrOfBlocklengths, arrOfDisplacements, &typeb);
        setType(localM, localK, arrOfTypes, arrOfBlocklengths, arrOfDisplacements, &typec);

        MPI_Type_free(&arrOfTypes[0]);
    }

    // 1. Нулевая ветвь передает (scatter) горизонтальные полосы матрицы A по x координате
    if (coords[1] == 0) {
        MPI_Scatter(A, localM * N, MPI_DOUBLE, partA, localM * N, MPI_DOUBLE, 0, comm_1D[0]);
    }

    // 2. Нулевая ветвь передает (scatter) горизонтальные полосы матрицы B по y координате
    if (coords[0] == 0) {
        // == разделение матрицы В вдоль координаты у
        // Вычисление размера подматрицы partB и смещений каждой подматрицы в матрице B.
        // Подматрицы partB упорядочены в B в соответствии с порядком номеров компьютеров в решетке, т.к.
        // массивы расположены в памяти по строкам, то подматрицы partB в памяти (в B) должны располагаться
        // в следующей последовательности: PartB0, PartB1,....
        // Смещения и размер подматриц CС для сборки в корневом процессе (ветви)
        int *dispb = (int *) malloc(p[1] * sizeof(int));
        int *countb = (int *) malloc(p[1] * sizeof(int));
        for (int j = 0; j < p[1]; j++) {
            dispb[j] = j;
            countb[j] = 1;
        }

        MPI_Scatterv(B, countb, dispb, typeb, partB,
                     N * localK, MPI_DOUBLE, 0, comm_1D[1]);
    }

    // 3. Передача подматриц partA в измерении y */
    MPI_Bcast(partA, localM * N, MPI_DOUBLE, 0,
              comm_1D[1]);

    // 4. Передача подматриц partB в измерении x */
    MPI_Bcast(partB, N * localK, MPI_DOUBLE, 0,
              comm_1D[0]);

    // 5. Вычисление подматриц PartС в каждой ветви */
    for (int i = 0; i < localM; i++) {
        for (int j = 0; j < localK; j++) {
            partC[localK * i + j] = 0.0;
            for (int k = 0; k < N; k++) {
                partC[localK * i + j] = partC[localK * i + j] + partA[N * i + k] * partB[localK * k + j];
            }
        }
    }

    // Вычисление размера подматрицы PartС и смещений каждой подматрицы в матрице C. Подматрицы partC
    // упорядочены в С в соответствии с порядком номеров компьютеров в решетке, т.к. массивы расположены в
    // памяти по строкам, то подматрицы PartС в памяти (в С) должны располагаться в следующей
    // последовательности: PartС0, PartС1, PartС2, PartC3, PartС4, PartС5, PartС6, PartС7.
    // Смещения и размер подматриц PartС для сборки в корневом процессе (ветви)
    int *dispc = (int *) malloc(p[0] * p[1] * sizeof(int));
    int *countc = (int *) malloc(p[0] * p[1] * sizeof(int));
    for (int i = 0; i < p[0]; i++) {
        for (int j = 0; j < p[1]; j++) {
            dispc[i * p[1] + j] = (i * p[1] * localM + j);
            countc[i * p[1] + j] = 1;
        }
    }

    // 6. Сбор всех подматриц PartС в ветви 0
    MPI_Gatherv(partC, localM * localK, MPI_DOUBLE, C,
                countc, dispc, typec, 0, comm_2D);

    free(partA);
    free(partB);
    free(partC);
    MPI_Comm_free(&comm_dup);
    MPI_Comm_free(&comm_2D);
    for (int & i : comm_1D) {
        MPI_Comm_free(&i);
    }
    if (rank == 0) {
        free(countc);
        free(dispc);
        MPI_Type_free(&typeb);
        MPI_Type_free(&typec);
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

    if (argc == 3) {
        p[0] = strtol(argv[1], nullptr, 10);
        p[1] = strtol(argv[2], nullptr, 10);
        if (p[0] * p[1] != size) {
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

        initABC(A, B, C, expected, M, N, K);

        getExpected(expected, A, B, N, M, K);
    }

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    MatrixMultiply_2D(A, B, C, p, comm);
    printRes(C, rank, N, M, K);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Processes: %d; Time: %f\n", size, end_time - start_time);
    }

    if (rank == 0) {
        if (areEqual(C, expected, N, M, K)) {
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