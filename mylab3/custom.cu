#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

__global__ void linear_forward_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < M && col < K){
        float value = 0.0f;
        for(int i=0;i<N;i++){
            value += A[row * N + i] * B[K * i + col];
        }
        C[row*K+col] = value + bias[col];
    }
}

__global__ void linear_backward_kernel(
    const float* grad_output, const float* input, const float* weights, 
    float* grad_input, float* grad_weights, float* grad_bias, int M, int N, int K){
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockDim.y + threadIdx.y;

    // cal grad_input (grad_output * weights^T)
    if(row < M && col < K){
        float value = 0.0f;
        for(int i=0;i<N;i++){
            value += grad_output[row*K +i] * weights[col*K +i];
        }
        grad_input[row*N+col] = value;
    }
    //cal grad_weights (input^T * grad_output)
    if(row < N && col < K){
        float value = 0.0f;
        for(int i=0;i<M;i++){
            value += input[i*N+row] * grad_output[i*K+col];
        }
        grad_weights[row*K+col] = value;
    }
    //cal grad_bias (the sum of grad_output's column)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < M; ++i) {
            atomicAdd(&grad_bias[col], grad_output[i * K + col]);
        }
    }     
} 