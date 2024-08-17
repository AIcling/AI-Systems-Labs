# LAB2 - [Custom Operators](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/README.md)

### Objective: Design and implement a custom operator (both forward and backward) in Python.

### Implementation Overview:

1. **PyTorch Implementation of a Custom Linear Function: [`custom_linear.py`](custom_linear.py)**

   The `CustomLinearFunction` class is a custom implementation of a linear transformation using PyTorch's `autograd.Function`:

   ```python
   class CustomLinearFunction(Function):
       @staticmethod
       def forward(ctx, input, weight, bias=None):
           ctx.save_for_backward(input, weight, bias)
           output = input.mm(weight.t())
           if bias is not None:
               output += bias
           return output

       @staticmethod
       def backward(ctx, grad_output):
           input, weight, bias = ctx.saved_tensors
           grad_input = grad_weight = grad_bias = None
           if ctx.needs_input_grad[0]:
               grad_input = grad_output.mm(weight)
           if ctx.needs_input_grad[1]:
               grad_weight = grad_output.t().mm(input)
           if bias is not None and ctx.needs_input_grad[2]:
               grad_bias = grad_output.sum(0)
           return grad_input, grad_weight, grad_bias
   ```

   A corresponding PyTorch module for using this custom function:

   ```python
   class CustomLinear(nn.Module):
       def __init__(self, in_features, out_features, bias=True):
           super(CustomLinear, self).__init__()
           self.in_features = in_features
           self.out_features = out_features
           self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
           self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
           self.reset_parameters()

       def reset_parameters(self):
           ...

       def forward(self, input):
           return CustomLinearFunction.apply(input, self.weight, self.bias)
   ```
2. **Custom Linear Function with C++ Extension: [`custom_linear.cpp`](custom_linear.cpp)**

   Implementation of a custom linear function in C++:

   ```cpp
   torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
   std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
   ```
3. **Custom Convolution Function with C++ Extension: [`custom_conv.cpp`](custom_conv.cpp)**

   Implementation of a custom 2D convolution function in C++:

   ```cpp
   torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding);
   std::vector<torch::Tensor> conv2d_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, int64_t stride, int64_t padding);
   ```
4. **Custom Dropout Function with C++ Extension: [`custom_dropout2d.cpp`](custom_dropout2d.cpp)**

   Implementation of a custom 2D dropout function in C++:

   ```cpp
   torch::Tensor dropout2d_forward(torch::Tensor input, double p, bool training);
   torch::Tensor dropout2d_backward(torch::Tensor grad_output, torch::Tensor mask, double p);
   ```
5. **Building the C++ Extensions:**

   You can build the C++ extensions using either **CMake** or **setup.py**.

   - **Using CMake:**

     ```bash
     mkdir build
     cd build
     cmake ..
     make
     ```
   - **Using `setup.py`:**

     ```bash
     python setup.py build_ext --inplace
     ```

### Project Structure:

```bash
.
├── README.md
├── data
│   └── MNIST
├── libtorch
│   ├── bin
│   ├── build-hash
│   ├── build-version
│   ├── include
│   ├── lib
│   └── share
└── mylab2
    ├── CMakeLists.txt
    ├── README.md
    ├── bindings.cpp
    ├── build
    ├── custom_conv.cpp
    ├── custom_cpp_extension.so
    ├── custom_dropout2d.cpp
    ├── custom_linear.cpp
    ├── custom_linear.py
    ├── example.py
    ├── mycpp_extension.cpython-311-x86_64-linux-gnu.so
    ├── setup.py
    └── test.py
```

This structure and detailed implementation guide should help in navigating the project more effectively.
