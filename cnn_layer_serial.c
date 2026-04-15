#include "cnn_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


void readMatrixFromFile(const char *filename, double ***matrix, int *size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s.\n", filename);
        exit(1);
    }

    // Read dimensions
    fscanf(file, "%d", size);
    //printf("Reading matrix from file: %s\n", filename);
    //printf("Dimensions: %d\n", *size);

    // Allocate memory for the matrix dynamically
    *matrix = (double **)malloc(*size * sizeof(double *));
    for (int i = 0; i < *size; i++) {
        (*matrix)[i] = (double *)malloc(*size * sizeof(double));
    }

    // Read matrix elements
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < *size; i++) {
        for (int j = 0; j < *size; j++) {
            fscanf(file, "%lf", &(*matrix)[i][j]);
            //printf("%.2lf\t", (*matrix)[i][j]);  // Print matrix elements as they are read
        }
        //printf("\n");
    }

    fclose(file);
}


void printMatrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2lf\t", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void addZeroPadding(double ***matrix, int *size) {
    int originalSize = *size;
    int newSize = originalSize + 2; // Increase size by 2 to add padding

    // Create a new matrix with zero padding
    double **paddedMatrix = (double **)malloc(newSize * sizeof(double *));
    for (int i = 0; i < newSize; i++) {
        paddedMatrix[i] = (double *)malloc(newSize * sizeof(double));
    }

    // Initialize the padded matrix with zeros
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            paddedMatrix[i][j] = 0.0;
        }
    }

    // Copy the original matrix to the center of the padded matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < originalSize; i++) {
        for (int j = 0; j < originalSize; j++) {
            paddedMatrix[i + 1][j + 1] = (*matrix)[i][j];
        }
    }

    // Update the size
    *size = newSize;

    // Free the original matrix
    for (int i = 0; i < originalSize; i++) {
        free((*matrix)[i]);
    }
    free(*matrix);

    // Assign the padded matrix to the original matrix pointer
    *matrix = paddedMatrix;
}

void freeMatrix(double **matrix, int size) {  // Updated function name
    // Implementation for freeing allocated memory
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void convolutionOperation(double **input, int inputSize, double **kernel, int kernelSize, double ***output) {
    // Calculate the size of the output matrix
    int outputSize = inputSize - kernelSize + 1;

    int num_threads = 12; // Set this value dynamically based on your requirements
    omp_set_num_threads(num_threads);


    // Allocate memory for the output matrix
    *output = (double **)malloc(outputSize * sizeof(double *));
    for (int i = 0; i < outputSize; i++) {
        (*output)[i] = (double *)malloc(outputSize * sizeof(double));
    }

    #pragma omp parallel for collapse(2) // Parallelize nested loops
    // Perform convolution
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // Compute the convolution sum for the current position
            double sum = 0.0;
            
            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            // Assign the sum to the corresponding position in the output matrix
            (*output)[i][j] = sum;
        }
    }
}

void maxPooling(double **input, int inputSize, double ***output) {
    // Calculate the size of the output matrix
    int outputSize = inputSize / 2;

    // Allocate memory for the output matrix
    *output = (double **)malloc(outputSize * sizeof(double *));
    for (int i = 0; i < outputSize; i++) {
        (*output)[i] = (double *)malloc(outputSize * sizeof(double));
    }

    // Perform max pooling
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // Find the maximum value in the 2x2 tile
            double maxVal = input[2*i][2*j];
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    if (input[2*i+k][2*j+l] > maxVal) {
                        maxVal = input[2*i+k][2*j+l];
                    }
                }
            }
            // Assign the maximum value to the output matrix
            (*output)[i][j] = maxVal;
        }
    }
}

void writeMatrixToFile(char *filename, double **matrix, int newSize) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s for writing.\n", filename);
        exit(1);
    }

    fprintf(file, "%d\n", newSize);
    #pragma omp parallel for 
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            fprintf(file, "%.8lf\t", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void applySigmoid(double **input, double **output, int newSize) {
    // Implementation for applying sigmoid function element-wise and filling the output matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            double exp_val = exp(-input[i][j]);
            double sigmoid_val = 1.0 / (1.0 + exp_val);
            //printf("input[%d][%d] = %lf, sigmoid_val = %lf\n", i, j, input[i][j], sigmoid_val);
            output[i][j] = sigmoid_val;
        }
    }
}

void addZeroPadding2(double ***matrix, int *size) {
    int originalSize = *size;
    int newSize = originalSize;
    // Check if the size is odd, then increase the new size and add padding
    if (originalSize % 2 != 0) {
        newSize++;
    }

    // Create a new matrix with zero padding
    double **paddedMatrix = (double **)malloc(newSize * sizeof(double *));
    
    for (int i = 0; i < newSize; i++) {
        paddedMatrix[i] = (double *)malloc(newSize * sizeof(double));
    }

    // Initialize the padded matrix with zeros
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            paddedMatrix[i][j] = 0.0;
        }
    }

    
    // Copy the original matrix to the top-left corner of the padded matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < originalSize; i++) {
        for (int j = 0; j < originalSize; j++) {
            paddedMatrix[i][j] = (*matrix)[i][j];
        }
    }

    // Update the size
    *size = newSize;

    // Free the original matrix
    
    for (int i = 0; i < originalSize; i++) {
        free((*matrix)[i]);
    }
    free(*matrix);

    // Assign the padded matrix to the original matrix pointer
    *matrix = paddedMatrix;
}
