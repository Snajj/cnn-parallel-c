#ifndef CNN_LAYER_H
#define CNN_LAYER_H

#include <stdbool.h>

void readMatrixFromFile(const char *filename, double ***matrix, int *rows);
void printMatrix(double **matrix, int size);
void addZeroPadding(double ***matrix, int *size);
void freeMatrix(double **matrix, int size);
void convolutionOperation(double **input, int inputSize, double **kernel, int kernelSize, double ***output);
void maxPooling(double **input, int inputSize, double ***output);
void writeMatrixToFile(char *filename, double **matrix, int newSize);
void applySigmoid(double **input, double **output, int newSize);
void addZeroPadding2(double ***matrix, int *size);

#endif  // CNN_LAYER_H
