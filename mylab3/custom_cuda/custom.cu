__global__ void matrixMulKernel(float* input, float* weights, float* output, int N, int inputDim, int H) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 对应于输入矩阵的行
    int col = blockIdx.y * blockDim.y + threadIdx.y; // 对应于输出矩阵的列

    if (row < N && col < H) {
        float value = 0.0;
        for (int k = 0; k < inputDim; ++k) { 
            value += input[row * inputDim + k] * weights[k * H + col];
        }
        output[row * H + col] = value;
    }
}

void matrixMul(float* input, float* weights, float* output, int N, int inputDim, int H) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);

    matrixMulKernel<<<gridDim, blockDim>>>(input, weights, output, N, inputDim, H);

    cudaDeviceSynchronize();
}
