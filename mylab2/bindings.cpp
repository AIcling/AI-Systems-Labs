#include <torch/extension.h>

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding);
std::vector<torch::Tensor> conv2d_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, int64_t stride, int64_t padding);

torch::Tensor dropout2d_forward(torch::Tensor input, double p, bool training);
torch::Tensor dropout2d_backward(torch::Tensor grad_output, torch::Tensor mask, double p);

PYBIND11_MODULE(custom_extension, m) {
    m.def("linear_forward", &linear_forward, "Custom Linear Forward");
    m.def("linear_backward", &linear_backward, "Custom Linear Backward");
    
    m.def("conv2d_forward", &conv2d_forward, "Custom Conv2D Forward");
    m.def("conv2d_backward", &conv2d_backward, "Custom Conv2D Backward");

    m.def("dropout2d_forward", &dropout2d_forward, "Custom Dropout2d Forward");
    m.def("dropout2d_backward", &dropout2d_backward, "Custom Dropout2d Backward");
}
