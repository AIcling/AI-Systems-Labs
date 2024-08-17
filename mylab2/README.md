# LAB2-[Customize operators](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/README.md)

### __Remark: Design and implement a customized operator (both forward and backward) in python__

### My implementation steps are as follows:

1. **a pytorch implemantation of customized linear function: ``custom_linear.py``**``

The `CustomLinearFunction` class is a custom implementation of a linear transformation using PyTorch's `autograd.Function`.

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

The `ctx` object is a flexible mechanism for sharing data between the forward and backward passes in a custom autograd function.

Then a pytorch implemention of linear function:

```python
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        ...

    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weight, self.bias)
```

2. **Customize a linear function with cpp extention**: ``custom_linear.cpp``

```cpp
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
```

3. **Customize a conv function with cpp extention**: ``custom_conv.cpp``

```cpp
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding);
std::vector<torch::Tensor> conv2d_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, int64_t stride, int64_t padding);
```

4. **Customize a dropout function with cpp extention**: ``custom_dropout2d.cpp``

   ```cpp
   torch::Tensor dropout2d_forward(torch::Tensor input, double p, bool training);
   torch::Tensor dropout2d_backward(torch::Tensor grad_output, torch::Tensor mask, double p);
   ```
5. Finally, it needs to be binded with cpp-pytorch extensions.

   One choice is to use cmake directly: for me, I make use of ``CMakeLists.txt`` to define the compiller.

   Please note that **the corresponding g++ version of Pytorch is a must.**

   follow the instruction:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
   Or you can use ``setup.py`` which is banefit from ``torch.utils.cpp_extension`` tools.

   follow the instruction:

   ```bash
   python setup.py build_ext --inplace
   ```
   ### My implemention results

   here is a general structure of my project:


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
