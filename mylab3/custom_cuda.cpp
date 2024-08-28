#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor linear_forward_cuda(torch::Tensor input, torch::Tensor weights, torch::Tensor bias){
    torch::Tensor output = torch::zeros({input.size(0),weights.size(1)},input.options());

    int M = input.size(0);
    int N = input.size(1);
    int K = weights.size(1);

    dim3 blocksize(16,16);
    dim3 gridsize((K+blocksize.x-1)/blocksize.x,(M+blocksize.y-1)/blocksize.y);

    linear_forward_kernel<<<gridsize,blocksize>>>(input.data_ptr<float>(),weights.data_ptr<float>(),
                                                bias.data_ptr<float>(),output.data_ptr<float>(),M,N,K);
    
    return output;
}

std::vector<torch::Tensor> linear_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weights) {
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_bias = torch::zeros({weights.size(1)}, grad_output.options());

    const int M = input.size(0);
    const int N = input.size(1);
    const int K = weights.size(1);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocksInput((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    dim3 numBlocksWeights((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linear_backward_kernel<<<numBlocksInput, threadsPerBlock>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(), weights.data_ptr<float>(), 
        grad_input.data_ptr<float>(), grad_weights.data_ptr<float>(), grad_bias.data_ptr<float>(), M, N, K);

    return {grad_input, grad_weights, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward_cuda", &linear_forward_cuda, "CUDA Linear Forward");
    m.def("linear_backward_cuda", &linear_backward_cuda, "CUDA Linear Backward");
}