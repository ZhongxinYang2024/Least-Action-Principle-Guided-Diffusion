import torch

def Uniform_upsample_3D(input_tensor, ratio):
    a,b,Tlen = input_tensor.shape
    n_split = Tlen - 1  # 原始点数量为 n_split+1
    Tlen_target = 4 * n_split + 1  # 目标长度
    
    if n_split == 0:
        # 当 n=0 时，直接返回原始 tensor（形状已是 [a,b,1]）
        return input_tensor
    
    # 初始化输出 tensor
    output = torch.zeros(a, b, Tlen_target, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # 生成目标位置索引 [0, 1, ..., 4*n]
    j = torch.arange(Tlen_target, device=input_tensor.device)
    
    # 计算原始索引和余数
    i = j // ratio  # 整数除法，floor(j/4) [0, 0, 0, 0, 1, 1, 1, 1, 2,...]
    r = (j % ratio).float()  # 余数，转换为 float 用于计算权重 [0.0, 1.0, 2.0, 3.0, 0.0, ...]
    
    # 计算权重：左权重 (1 - r/4)，右权重 (r/4)
    w_left = 1 - r / float(ratio) # [1.0, 0.75, 0.5, 0.25, 1.0, ...]
    w_right = r / float(ratio)    # [0.0, 0.25, 0.5, 0.75, 0.0, ...]
    
    # 处理前 4*n 个点（使用线性插值）
    # 左点: input[..., i]，右点: input[..., i+1]
    output[..., :-1] = w_left[:-1] * input_tensor[..., i[:-1]] + w_right[:-1] * input_tensor[..., i[:-1] + 1]
    
    # 处理最后一个点（直接使用原始最后一个值）
    output[..., -1] = input_tensor[..., -1]
    
    return output

if __name__ == '__main__':
    # 示例输入：2x2x3 (n=2)
    input_tensor = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ])

    output = Uniform_upsample_3D(input_tensor, 4)
    print(output.shape)  # 输出: torch.Size([2, 2, 9])