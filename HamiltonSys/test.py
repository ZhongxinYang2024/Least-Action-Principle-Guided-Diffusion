import torch
import torch.nn as nn

# 定义被积函数 f(t)
def f(t):
    return t**2 + 3*t + 1  # 示例函数，可以替换为任意函数

# 创建计算图
x = torch.tensor(2.0, requires_grad=True)  # 定义变量 x，需要求导

# 使用数值积分计算 ∫₀ˣ f(t) dt
# 这里使用梯形法则进行数值积分
def integral_from_0_to_x(x, n=1000):
    t_values = torch.linspace(0, x.item(), n, requires_grad=False)
    integral = torch.trapz(f(t_values), t_values)
    return integral

# 计算积分
integral_value = integral_from_0_to_x(x)

# 使用自动微分计算导数
integral_value.backward()

# 输出结果
print(f"F({x.item()}) = ∫₀^{x.item()} f(t) dt = {integral_value.item()}")
print(f"F'({x.item()}) = d(∫₀^{x.item()} f(t) dt)/dx = {x.grad.item()}")
print(f"根据微积分基本定理，f({x.item()}) = {f(x).item()}")