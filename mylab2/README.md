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
