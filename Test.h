//
// Created by Admin on 05.05.2020.
//

#ifndef OPPLAB3_TEST_H
#define OPPLAB3_TEST_H

void getExpected(double *result, double *matrix1, double *matrix2, int sizeN, int sizeM, int sizeK);
bool areEqual(double *C, double *expectedC, int sizeN, int sizeM, int sizeK);
void printRes(double *C, int MyP, int sizeN, int sizeM, int sizeK);
void printExpected(double *result);

#endif //OPPLAB3_TEST_H
