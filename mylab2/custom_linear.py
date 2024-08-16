import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 自定义前向和反向传播的函数
class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # 保存变量供反向传播使用
        ctx.save_for_backward(input, weight, bias)
        
        # 计算前向传播
        output = input.mm(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文中检索保存的变量
        input, weight, bias = ctx.saved_tensors
        
        # 计算梯度
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias

# 基于 nn.Module 实现自定义的 Linear 类
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
        
        # 初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weight, self.bias)
    

class cutom_Net1(nn.Module):
    def __init__(self):
        super(cutom_Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        self.fc1 = CustomLinear(9216, 128)  
        self.fc2 = CustomLinear(128, 10)    

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output