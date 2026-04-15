# Parallel CNN Layer Implementation in C (OpenMP)

## Overview

This project implements a simplified CNN-style computation pipeline in C, including convolution, activation, and pooling operations. The focus is on performance optimization using OpenMP and understanding the trade-offs of parallel execution on a multi-core CPU.

The program processes an input matrix with multiple kernels and generates corresponding output feature maps.

---

## Features

- 2D convolution implemented from scratch (stride = 1)
- Zero padding for boundary handling
- Sigmoid activation (element-wise)
- Max pooling (2x2, stride = 2)
- Dynamic memory allocation for matrix operations
- File-based input/output system
- OpenMP-based parallelization
- Performance analysis using gprof and omp_get_wtime

---

## Workflow

The program:

1. Reads the input matrix
2. Processes each kernel (kernel1, kernel2, kernel3)
3. Applies:
   - convolution
   - sigmoid activation
   - optional padding
   - max pooling
4. Writes outputs to corresponding files

---

## Build

make

---

## Run

./main

---

## Parallelization Strategy

Two levels of parallelization were explored:

Intra-kernel parallelism (effective):
Parallelized nested loops inside convolution using OpenMP.
Example: 
```c
#pragma omp parallel for collapse(2)
```
This provided a significant speedup (approximately 3x depending on thread count).

Inter-kernel parallelism (not effective):
Attempted to run multiple kernels in parallel.
This resulted in slower performance due to:
- thread oversubscription
- memory contention
- CPU resource limits

---

## Performance Insights

- Convolution is the main computational bottleneck
- Parallelizing inner loops is more effective than parallelizing high-level tasks
- Increasing thread count improves performance up to hardware limits
- Over-parallelization can degrade performance on consumer CPUs

Example observations:

- Serial execution: ~4.5 seconds
- Parallel convolution: ~1.3–1.5 seconds
- Fully parallel main execution: up to ~40 seconds (slower)

---

## System

Tested on a 6-core / 12-thread CPU Ryzen 5 5600H CPU.

---

## Notes

- Uses square matrices for simplicity
- Designed as a low-level implementation rather than using ML frameworks

---

## Author

Mert Kaya
