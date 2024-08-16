# LAB2-定制一个新的张量运算

### 具体步骤

1. 在MNIST的模型样例中，选择线性层（Linear）张量运算进行定制化实现
2. 理解PyTorch构造张量运算的基本单位：Function和Module
3. 基于Function和Module的Python API重新实现Linear张量运算

   1. 修改MNIST样例代码
   2. 基于PyTorch  Module编写自定义的Linear 类模块
   3. 基于PyTorch Function实现前向计算和反向传播函数
   4. 使用自定义Linear替换网络中nn.Linear() 类
   5. 运行程序，验证网络正确性
4. 理解PyTorch张量运算在后端执行原理
5. 实现C++版本的定制化张量运算

   1. 基于C++，实现自定义Linear层前向计算和反向传播函数，并绑定为Python模型
   2. 将代码生成python的C++扩展
   3. 使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数
   4. 运行程序，验证网络正确性
6. 使用profiler比较网络性能：比较原有张量运算和两种自定义张量运算的性能
7. 【可选实验，加分】实现卷积层（Convolutional）的自定义张量运算我的实现过程

### 我的实现

pytorch实现：`custom_linear.py`

PyTorch 的张量运算在后端执行的原理涉及：**自动求导机制**和**动态计算图**。

* **`ctx`** 是 PyTorch 自动生成的context对象，它在前向传播时被传递给 `forward` 方法，并在反向传播时传递给 `backward` 方法。可以使用 `ctx` 来存储任意信息，以便在反向传播时使用。
* **`ctx.save_for_backward(*tensors)`** 方法可以保存前向传播中的张量，这些张量将在反向传播中使用。这些张量会被存储在反向传播过程中 `ctx.saved_tensors` 中。

前向传播的执行过程：

1. **调用 `forward` 方法**: 当执行 `CustomLinear` 的前向传播时，PyTorch 会调用 `forward` 方法，该方法的核心是 `CustomLinearFunction.apply`：这里的 `apply` 方法是 PyTorch `autograd.Function` 类的标准方法，它用于执行自定义的前向传播和记录反向传播所需的信息。
2. **进入 `CustomLinearFunction` 的 `forward` 静态方法**: 在调用 `apply` 后，控制权转移到 `CustomLinearFunction.forward`：在这个 `forward` 方法中，执行了如下步骤：**保存中间变量**（`ctx` 是上下文对象，通过 `ctx.save_for_backward` 保存了输入张量 `input`、权重 `weight` 和偏置 `bias`。这些变量将在反向传播时使用。）；计算输出（这一步执行矩阵乘法，将输入张量与转置后的权重矩阵相乘，得到输出张量 `output`。这是线性层的核心操作。）最后，`forward` 方法返回计算得到的输出张量。
3. **返回到模型的前向传播**: 经过 `CustomLinearFunction` 处理后，计算得到的输出会返回给 `CustomLinear` 模块的 `forward` 方法，并继续传递到模型的下一层。
   ```
       def forward(ctx, input, weight, bias=None):
           ctx.save_for_backward(input, weight, bias)
           output = input.mm(weight.t())
           if bias is not None:
               output += bias
           return output
   ```
