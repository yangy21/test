import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_fc = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel//ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel//ratio, out_channels=in_channel, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_fc(self.avg_pool(x))
        max_out = self.shared_fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out



class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                               padding=7//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道方向压缩
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))

        return out


class CBAM_Module(nn.Module):
    def __init__(self, in_channel):
        super(CBAM_Module, self).__init__()
        self.channel_attention = ChannelAttention(in_channel=in_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


if __name__ == '__main__':
    model = CBAM_Module(in_channel=64)
    data = torch.randn([1, 64, 32, 32]).float()
    out = model(data)
    print(out.shape)
















# model = ChannelAttention(in_channel=64)
# data = torch.randn([1, 64, 32, 32]).float()
# out = model(data)
# print('out.shape:', out.shape)

# layer = nn.AdaptiveAvgPool2d(1)
# #layer = nn.AvgPool2d(1)
# data = torch.tensor([[1,2,3],
#                      [4,5,6],
#                      [7,8,9]]).float()
# print(data.shape)
# out = layer(data)
# print(out)
# print(out.shape)