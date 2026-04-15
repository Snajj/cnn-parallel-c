# cnn-parallel-c
OpenMP-optimized CNN layer implementation in C

# Parallel CNN Layer Implementation in C (OpenMP)

## Overview

This project implements a simplified CNN-style computation pipeline in C, including convolution, activation, and pooling operations. The focus of the project is on performance optimization using OpenMP and understanding the trade-offs of parallel execution on a multi-core CPU.

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
- Performance analysis using `gprof` and `omp_get_wtime`

---

The program will:

Read input matrix
Process each kernel (kernel1, kernel2, kernel3)
Apply:
convolution
sigmoid activation
optional padding
max pooling
Write outputs to corresponding files
Parallelization Strategy

Two levels of parallelization were explored:

1. Intra-kernel parallelism (effective) Provided significant speedup (≈3x depending on thread count)
Parallelized nested loops inside convolution using: #pragma omp parallel for collapse(2)

2. Inter-kernel parallelism (not effective)
Attempted to run multiple kernels in parallel
Resulted in slower performance due to:
thread oversubscription
memory contention
CPU resource limits
Performance Insights
Convolution is the main computational bottleneck
Parallelizing inner loops is more effective than parallelizing high-level tasks
Increasing thread count improves performance up to hardware limits
Over-parallelization can degrade performance on consumer CPUs

Example observations:

Serial execution: ~4.5s
Parallel convolution: ~1.3–1.5s
Fully parallel main execution: up to ~40s (worse)
