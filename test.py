import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleImageEncoder(nn.Module):
    """
    一个简单的CNN编码器，将输入图像编码为固定维度的特征向量
    
    参数:
        input_channels (int): 输入图像的通道数，默认为3(RGB)
        base_channels (int): 基础通道数，控制网络容量，默认为64
        latent_dim (int): 输出特征向量的维度，默认为512
    """
    def __init__(self, input_channels=3, base_channels=64, latent_dim=512):
        super(SimpleImageEncoder, self).__init__()
        
        # 卷积层序列
        self.conv_layers = nn.Sequential(
            # 第一层: 输入 -> base_channels
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 第二层: base_channels -> base_channels*2
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            # 第三层: base_channels*2 -> base_channels*4
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            
            # 第四层: base_channels*4 -> base_channels*8
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化 (将空间维度降为1x1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层，将特征映射到潜在空间
        self.fc = nn.Linear(base_channels*8, latent_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入图像，形状为 [batch_size, input_channels, H, W]
        
        返回:
            Tensor: 特征向量，形状为 [batch_size, latent_dim]
        """
        # 通过卷积层提取特征
        x = self.conv_layers(x)
        
        # 全局平均池化，将每个通道的空间信息压缩为一个值
        x = self.global_avg_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 通过全连接层生成特征向量
        feature_vector = self.fc(x)
        
        return feature_vector

# 使用示例
if __name__ == "__main__":
    # 创建一个编码器实例
    encoder = SimpleImageEncoder(input_channels=3, base_channels=64, latent_dim=512)
    
    # 创建一个随机输入图像 (batch_size=4, channels=3, height=128, width=128)
    dummy_input = torch.randn(4, 3, 56, 56)
    
    # 前向传播
    feature_vector = encoder(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出特征向量形状: {feature_vector.shape}")  # 应该是 [4, 512]