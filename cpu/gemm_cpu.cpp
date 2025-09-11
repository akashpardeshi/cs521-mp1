#include "../include/utils.h"
#include <chrono>

#define NUM_RUNS 4

#define CHECK(name)                                                            \
  std::cout << "checking " << #name << std::endl;                              \
  initialize(refC, Ref::M *Ref::N);                                            \
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);                            \
  if (!ref.checkRef(refC)) {                                                   \
    std::cerr << #name << ": check ref failed!" << std::endl;                  \
  };

#define TIME(name)                                                             \
  for (int i = 0; i < 1; i++) {                                                \
    name(A, B, C, M, N, K);                                                    \
  }                                                                            \
  std::chrono::duration<double, std::milli> time_##name(0);                    \
  for (int i = 0; i < NUM_RUNS; i++) {                                         \
    initialize(C, M * N);                                                      \
    auto start_time_##name = std::chrono::high_resolution_clock::now();        \
    name(A, B, C, M, N, K);                                                    \
    auto end_time_##name = std::chrono::high_resolution_clock::now();          \
    time_##name += end_time_##name - start_time_##name;                        \
  }                                                                            \
  std::chrono::duration<double, std::milli> duration_##name =                  \
      time_##name / float(NUM_RUNS);                                           \
  std::cout << "Time taken for GEMM (CPU," << #name                            \
            << "): " << duration_##name.count() << "ms" << std::endl;

void gemm_cpu_o0(float *A, float *B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void gemm_cpu_o1(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

#define TSIZE 32
void gemm_cpu_o2(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int kk = 0; kk < K; kk += TSIZE) {
      const int max_k = std::min(kk + TSIZE, K);
      for (int jj = 0; jj < N; jj += TSIZE) {
        const int max_j = std::min(jj + TSIZE, N);
        for (int k = kk; k < max_k; k++) {
          for (int j = jj; j < max_j; j++) {
            C[i * N + j] += A[i * K + k] * B[k * N + j];
          }
        }
      }
    }
  }
}

void gemm_cpu_o3(float *A, float *B, float *C, int M, int N, int K) {
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int kk = 0; kk < K; kk += TSIZE) {
      const int max_k = std::min(kk + TSIZE, K);
      for (int jj = 0; jj < N; jj += TSIZE) {
        const int max_j = std::min(jj + TSIZE, N);
        for (int k = kk; k < max_k; k++) {
          for (int j = jj; j < max_j; j++) {
            C[i * N + j] += A[i * K + k] * B[k * N + j];
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
    return 1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  float *A = new float[M * K]();
  float *B = new float[K * N]();
  float *C = new float[M * N]();

  fillRandom(A, M * K);
  fillRandom(B, K * N);

  // Check if the kernel results are correct
  // note that even if the correctness check fails all optimized kernels will
  // run. We are not exiting the program at failure at this point. It is a good
  // idea to add more correctness checks to your code. We may (at discretion)
  // verify that your code is correct.
  float *refC = new float[Ref::M * Ref::N]();
  auto ref = Ref();
  CHECK(gemm_cpu_o0)
  CHECK(gemm_cpu_o1)
  CHECK(gemm_cpu_o2)
  CHECK(gemm_cpu_o3)
  delete[] refC;

  TIME(gemm_cpu_o0)
  TIME(gemm_cpu_o1)
  TIME(gemm_cpu_o2)
  TIME(gemm_cpu_o3)

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
