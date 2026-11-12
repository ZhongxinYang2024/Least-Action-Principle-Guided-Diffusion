import torch


def compute_grad(f, x, *args, **kwargs):
    """
    计算函数 f(x, *args, **kwargs) 关于 x 的梯度 ∇x f(x, *args, **kwargs)
    
    参数:
        f: 可调用的函数
        x: 输入张量，需要计算梯度
        args, kwargs: 函数的其他参数
        
    返回:
        grad_x: ∇x f(x, *args, **kwargs)，形状与 x 相同
    """
    # 确保 x 需要梯度
    x_input = x.clone().detach().requires_grad_(True)
    
    # 计算函数值
    f_value = f(x_input, *args, **kwargs)
    
    # 检查 f_value 是否为标量
    if f_value.ndim == 0:
        # 标量输出 - 直接反向传播
        f_value.backward()
        grad_x = x_input.grad
    else:
        # 非标量输出 - 对每个输出分量独立计算梯度
        # 创建与 f_value 同形状的全1张量
        grad_outputs = torch.ones_like(f_value, device=f_value.device)
        
        # 计算梯度
        grad_x = torch.autograd.grad(
            outputs=f_value,
            inputs=x_input,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
    
    return grad_x


def test_compute_grad_batch():
    import torch
    import torch.nn as nn

    # 测试配置
    batch_size = 5
    input_dim = 3
    output_dim = 1
    
    # 创建模型和输入
    model = nn.Linear(input_dim, output_dim)
    x = torch.randn(batch_size, input_dim)  # 批量输入
    
    # 测试1: 标量输出函数 (整个batch的损失值)
    def scalar_func(x):
        outputs = model(x)
        return outputs.pow(2).mean()  # 返回标量损失
    
    grad_scalar = compute_grad(scalar_func, x)
    
    # 验证标量输出梯度
    assert grad_scalar.shape == x.shape, "标量输出梯度形状错误"
    
    # 使用PyTorch自动微分验证
    x.requires_grad_(True)
    loss = scalar_func(x)
    loss.backward()
    torch_grad_scalar = x.grad
    assert torch.allclose(grad_scalar, torch_grad_scalar, atol=1e-6), "标量输出梯度值错误"
    
    print("测试1通过: 标量输出梯度正确")
    
    # 测试2: 非标量输出函数 (每个样本一个输出)
    def vector_func(x):
        return model(x)  # 返回形状为 [batch_size, output_dim] 的张量
    
    grad_vector = compute_grad(vector_func, x)
    
    # 验证非标量输出梯度
    assert grad_vector.shape == x.shape, "非标量输出梯度形状错误"
    
    # 使用PyTorch自动微分验证
    x.requires_grad_(True)
    outputs = vector_func(x)
    grad_outputs = torch.ones_like(outputs)
    torch_grad_vector = torch.autograd.grad(
        outputs, x, grad_outputs=grad_outputs, create_graph=False
    )[0]
    assert torch.allclose(grad_vector, torch_grad_vector, atol=1e-6), "非标量输出梯度值错误"
    
    print("测试2通过: 非标量输出梯度正确")
    
if __name__ == '__main__':
    test = 3;
    
    match test:
        case 1:
            # 示例 1: 标量输出函数
            def f_scalar(x):
                # 返回标量值 (batch_size 被求和)
                return torch.sum(x**2)

            # 示例 2: 批量输出函数
            def f_vector(x):
                # 返回每个样本的结果 (形状 [batch_size])
                return x[:, 0]**2 + 3*x[:, 1]

            # 批量输入（batch_size=3）
            x_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # 计算标量函数的梯度
            grad_scalar = compute_grad(f_scalar, x_batch)
            print("标量函数梯度:")
            print(grad_scalar)  # 应为 [2x, 2y] 的批量版本

            # 计算向量函数的梯度
            grad_vector = compute_grad(f_vector, x_batch)
            print("\n向量函数梯度:")
            print(grad_vector)  # 应为 [[2x, 3], ...] 的批量版本
        case 2:
            def Compute_Saction(x, x_00, x_neg1, T):
                """
                计算轨迹的速度和动能
                
                参数:
                x: 中间时刻位置张量, 形状 (batch_size, num_steps, spatial_dim)
                x_00: 初始位置张量, 形状 (batch_size, spatial_dim)
                x_neg1: 终态位置张量, 形状 (batch_size, spatial_dim)
                T: 总时间
                
                返回:
                Saction: 作用量张量, 形状 (batch_size,)
                """
                # 确保所有输入都是张量
                assert all(isinstance(t, torch.Tensor) for t in [x, x_00, x_neg1]), "所有输入必须是torch.Tensor"
                
                # 将初始和终态位置拼接到轨迹中
                full_x = torch.cat([
                    x_00.unsqueeze(1),  # 添加时间步维度并拼接在开头
                    x,
                    x_neg1.unsqueeze(1)  # 添加时间步维度并拼接在结尾
                ], dim=1)
                
                # 计算总时间步数 (包括初始和终态)
                total_steps = full_x.shape[1]
                
                # 计算时间间隔 (总时间除以步数间隔)
                dt = T / (total_steps - 1)  # 时间步长
                
                # 修复的关键：避免使用 torch.zeros_like 创建新张量
                # 而是使用 full_x 的差分直接计算速度
                
                # 计算前向差分：x_{t+1} - x_t
                forward_diff = full_x[:, 1:] - full_x[:, :-1]
                
                # 计算中心差分：x_{t+1} - x_{t-1}
                center_diff = full_x[:, 2:] - full_x[:, :-2]
                
                # 初始化速度张量
                v = torch.empty_like(full_x)
                
                # 边界点使用单侧差分
                v[:, 0] = forward_diff[:, 0] / dt  # 起始点 (前向差分)
                v[:, -1] = forward_diff[:, -1] / dt  # 结束点 (后向差分)
                
                # 内部点使用中心差分
                if total_steps > 2:
                    v[:, 1:-1] = center_diff / (2 * dt)
                
                # 计算动能 (假设质量为1)
                Kenergy = 0.5 * v ** 2
                
                # 计算势能
                Penergy = 0.5 * full_x ** 2
                
                # 计算作用量 (拉格朗日量在时间上的积分)
                Saction = torch.sum((Kenergy - Penergy) * dt, dim=1)
                
                return Saction


            def CondScore(x,T):
                Batch_size = x.shape[0]

                x_00 = torch.ones((Batch_size,),dtype=torch.float32, device=x.device)*0.0
                x_neg1 = torch.ones((Batch_size, ), dtype=torch.float32, device=x.device)*0.0
                Cgrad = compute_grad(Compute_Saction, x, x_00, x_neg1, T)
                return Cgrad

            x = torch.ones(16,2)
            grad_conditional = CondScore(x,1)
            print(grad_conditional[0])
        case 3:
            test_compute_grad_batch()
