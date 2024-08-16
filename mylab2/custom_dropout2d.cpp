#include <torch/extension.h>
#include <vector>
#include <random>

torch::Tensor generate_bernoulli_mask(torch::Tensor input, double p) {
    auto sizes = input.sizes();
    torch::Tensor mask = torch::zeros(sizes, input.options());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(1 - p);  

    auto mask_data = mask.data_ptr<float>();  
    int64_t numel = mask.numel();  

    for (int64_t i = 0; i < numel; ++i) {
        mask_data[i] = d(gen) ? 1.0f : 0.0f; 
    }

    return mask;
}

torch::Tensor dropout2d_forward(
    torch::Tensor input,
    double p,  
    bool training) {
    
    if (!training) {
        return input;
    }

    auto mask = generate_bernoulli_mask(input, 1 - p);   
    auto output = input * mask / (1 - p);  // 按概率保留并缩放
    
    return output;
}

torch::Tensor dropout2d_backward(
    torch::Tensor grad_output,
    torch::Tensor mask,
    double p) {
    
    return grad_output * mask / (1 - p);
}
