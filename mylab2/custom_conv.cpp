#include <torch/extension.h>
#include <vector>

// 自定义卷积层的前向传播实现
torch::Tensor custom_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {
    
    // 获取输入和权重的尺寸
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);

    // 计算输出尺寸
    auto output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // 初始化输出张量
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    // 为输入添加 padding
    if (padding > 0) {
        input = torch::constant_pad_nd(input, {padding, padding, padding, padding}, 0);
    }

    // 执行卷积操作
    for (int64_t n = 0; n < batch_size; ++n) {          // 遍历每个 batch
        for (int64_t oc = 0; oc < out_channels; ++oc) { // 遍历每个输出通道
            for (int64_t h = 0; h < output_height; ++h) {   // 遍历输出张量的高度
                for (int64_t w = 0; w < output_width; ++w) { // 遍历输出张量的宽度
                    // 计算对应的输入张量的区域
                    int64_t h_start = h * stride;
                    int64_t w_start = w * stride;
                    int64_t h_end = h_start + kernel_height;
                    int64_t w_end = w_start + kernel_width;

                    // 进行卷积运算
                    for (int64_t ic = 0; ic < in_channels; ++ic) { // 遍历每个输入通道
                        auto input_slice = input[n][ic].slice(0, h_start, h_end).slice(1, w_start, w_end);
                        auto weight_slice = weight[oc][ic];
                        output[n][oc][h][w] += (input_slice * weight_slice).sum().item<float>();
                    }

                    // 添加偏置
                    if (bias.defined()) {
                        output[n][oc][h][w] += bias[oc].item<float>();
                    }
                }
            }
        }
    }

    return output;
}

// 自定义卷积层的反向传播实现
std::vector<torch::Tensor> custom_conv2d_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding) {
    
    // 获取输入和权重的尺寸
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);

    // 初始化梯度张量
    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros(out_channels, input.options());

    // 为输入添加 padding
    if (padding > 0) {
        input = torch::constant_pad_nd(input, {padding, padding, padding, padding}, 0);
    }

    // 计算输入的梯度
    for (int64_t n = 0; n < batch_size; ++n) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            for (int64_t h = 0; h < grad_output.size(2); ++h) {
                for (int64_t w = 0; w < grad_output.size(3); ++w) {
                    int64_t h_start = h * stride;
                    int64_t w_start = w * stride;
                    int64_t h_end = h_start + kernel_height;
                    int64_t w_end = w_start + kernel_width;

                    for (int64_t ic = 0; ic < in_channels; ++ic) {
                        auto input_slice = input[n][ic].slice(0, h_start, h_end).slice(1, w_start, w_end);
                        auto grad_out_value = grad_output[n][oc][h][w];
                        grad_input[n][ic].slice(0, h_start, h_end).slice(1, w_start, w_end) += grad_out_value * weight[oc][ic];
                        grad_weight[oc][ic] += grad_out_value * input_slice;
                    }

                    grad_bias[oc] += grad_output[n][oc][h][w];
                }
            }
        }
    }

    // 去除 padding 对输入梯度的影响
    if (padding > 0) {
        grad_input = grad_input.slice(2, padding, -padding).slice(3, padding, -padding);
    }

    return {grad_input, grad_weight, grad_bias};
}

// 注册
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &custom_conv2d_forward, "Custom Conv2D Forward");
    m.def("conv2d_backward", &custom_conv2d_backward, "Custom Conv2D Backward");
}
