#include <torch/extension.h>
#include <vector>

// 前向计算
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto output = torch::matmul(input, weight.t());
    if (bias.defined()) {
        output += bias;
    }
    return output;
}

// 反向传播
std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto grad_input = torch::matmul(grad_output, weight);
    auto grad_weight = torch::matmul(grad_output.t(), input);
    torch::Tensor grad_bias = torch::Tensor();
    if (bias.defined()) {
        grad_bias = grad_output.sum(0);
    }
    return {grad_input, grad_weight, grad_bias};
}

// 定义模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward (C++)");
    m.def("linear_backward", &linear_backward, "Linear backward (C++)");
}
