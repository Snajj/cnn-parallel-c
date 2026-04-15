#include <stdio.h>
#include <stdlib.h>
#include "cnn_layer.h"
#include <math.h>
#include <omp.h>

// Function to process a kernel file and perform calculations
void processKernel(const char *kernelFileName) {
    int size1; 
    int size2; 
    int newSize;

    double **matrixA;
    double **matrixB;
    double **outputMatrix;
    double **outputMatrix2;

    // Read matrixA from input.txt
    double start; 
    double end; 
    start = omp_get_wtime(); 
    readMatrixFromFile("input.txt", &matrixA, &size1);
    end = omp_get_wtime(); 
    printf("readMatrixFromFile Matrix took %f seconds\n", end - start);
    double matrix_read = end - start;
    //readMatrixFromFile("random_matrix.txt", &matrixA, &size1);

    // Read matrixB from the given kernel file
    start = omp_get_wtime();
    readMatrixFromFile(kernelFileName, &matrixB, &size2);
    end = omp_get_wtime(); 
    printf("readMatrixFromFile Kernel took %f seconds\n", end - start);
    double kernel_read = end - start;
    // Add zero padding to matrixA
    ////printf("Zero Padding started\n");
    start = omp_get_wtime();
    addZeroPadding(&matrixA, &size1);
    end = omp_get_wtime(); 
    printf("Zero Padding took %f seconds\n", end - start);   
    double zero_padding = end - start;
    //printMatrix(matrixA, size1);
    //printf("Zero Padding done\n");
    // Perform convolution operation

    ////printf("Convolution Operation started\n");
    start = omp_get_wtime();
    convolutionOperation(matrixA, size1, matrixB, size2, &outputMatrix);
    end = omp_get_wtime(); 
    printf("Convolution took %f seconds\n", end - start);      
    double convolution = end - start;
    
    newSize = size1 - size2 + 1;
    //printMatrix(outputMatrix, newSize);

    // Apply sigmoid function to outputMatrix
    ////printf("Sigmoid Application started\n");
    start = omp_get_wtime();
    applySigmoid(outputMatrix, outputMatrix, newSize);
    end = omp_get_wtime(); 
    printf("Applying Sigmoid took %f seconds\n", end - start); 
    double sigmoid = end - start;
    //printf("Sigmoid Application done\n");
    ////printf("Current Size %d\n", newSize);
    //printMatrix(outputMatrix, newSize);

    // Add zero padding if needed
    if (newSize % 2 != 0) {
        ////printf("Additional Zero Padding started\n");
        addZeroPadding2(&outputMatrix, &newSize);
        //printf("Additional Zero Padding ended\n");
        //printMatrix(outputMatrix, newSize);
    }

    // Perform max pooling
    ////printf("Max Pooling started\n");
    start = omp_get_wtime();
    maxPooling(outputMatrix, newSize, &outputMatrix2);
    double output = end - start;
    end = omp_get_wtime(); 
    printf("Max Pooling took %f seconds\n", end - start); 
    //printf("Max Pooling ended\n");
    //printf("Current Size %d\n", newSize);
    
    
    newSize = newSize / 2; // Update newSize after pooling
    //printMatrix(outputMatrix2, newSize);

    // Write the output matrix to a new file
    char outputFileName[100]; // Adjust the size according to your needs
    sprintf(outputFileName, "Output_%s", kernelFileName); // Output file name

    ////printf("Writing output file started\n");
    start = omp_get_wtime();
    writeMatrixToFile(outputFileName, outputMatrix2, newSize);
    end = omp_get_wtime(); 
    printf("Writing file took %f seconds\n", end - start); 
    double file_write = end - start;
    ////printf("Output file written\n");

    // Free allocated memory
    freeMatrix(matrixA, size1);
    freeMatrix(matrixB, size2);
    freeMatrix(outputMatrix, newSize);
    freeMatrix(outputMatrix2, newSize);

    double total_time = matrix_read + kernel_read + zero_padding + convolution + sigmoid + output + file_write;
    printf("Total time passed: %f \n", total_time); 
    double convol_ratio = (convolution / total_time)*100;
    printf("Ratio of convolution to the rest %f \n", convol_ratio); 
}

/* int main() {
    double start; 
    double end; 
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        switch (i) {
            case 0:
                processKernel("kernel1.txt");
                break;
            case 1:
                processKernel("kernel2.txt");
                break;
            case 2:
                processKernel("kernel3.txt");
                break;
            default:
                printf("something broke :(");
                break;
        }
    }
    printf("Whole kernel took %f seconds\n", end - start); 
    return 0;
}
*/
/*
int main() {
    double start; 
    double end; 
    start = omp_get_wtime();
    
   // Process kernel1.txt
    processKernel("kernel1.txt");
 
    
    // Process kernel2.txt
    processKernel("kernel2.txt");
    
    // Process kernel3.txt
    processKernel("kernel3.txt");

    end = omp_get_wtime(); 
    printf("Whole kernel took %f seconds\n", end - start); 
    
    return 0;
}
*/
//HERE UNDERNEATH THE CODE FOR RUNNING ALL OF THE PROCESSKERNEL FUNCTION CALLS IN PARALLEL

int main() {
    double start, start2, start3; 
    double end, end2, end3; 
    int num_threads = 1; // Set this value dynamically based on your requirements
    omp_set_num_threads(num_threads);
    #pragma omp parallel 
    {
        // Each thread will execute one of the processKernel calls
        #pragma omp sections
        {
            #pragma omp section

            start = omp_get_wtime();
            processKernel("kernel1.txt");
            end = omp_get_wtime(); 
            printf("Kernel1 took %f seconds\n", end - start); 
            
            #pragma omp section
            start2 = omp_get_wtime();
            processKernel("kernel2.txt");
            end2 = omp_get_wtime(); 
            printf("Kernel2 took %f seconds\n", end2 - start2); 

            #pragma omp section
            start3 = omp_get_wtime();
            processKernel("kernel3.txt");
            end3 = omp_get_wtime(); 
            printf("Kernel3 took %f seconds\n", end3 - start3); 
        }
    }

}
