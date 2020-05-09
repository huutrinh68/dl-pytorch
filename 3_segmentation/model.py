import torch 
import torch.nn as nn


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # block 1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # block 2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # block 3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # block 4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)

        return outputs


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSP
        self.add_module("block1", bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        for i in range(n_blocks -1):
            self.add_module("block" + str(i+2), bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))




if __name__ == "__main__":
    x = torch.randn(1, 3, 475, 475)
    feature_conv = FeatureMap_convolution()
    outputs = feature_conv(x)
    print(outputs.shape) #torch.Size([1, 128, 119, 119])

