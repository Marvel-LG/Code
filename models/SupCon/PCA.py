import torch
import torch.nn as tnn
import torch.nn as nn


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.ReLU()
    )
    return layer


class VGG(tnn.Module):
    def __init__(self, in_channle, num_components=768):
        super(VGG, self).__init__()

        # Conv blocks (ReLU activation added in each block)
        self.layer1 = vgg_conv_block([in_channle], [in_channle * 2], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([in_channle * 2], [in_channle * 2], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([in_channle * 2], [in_channle * 2], [3, 3], [1, 1], 2, 2)
        self.layer4 = vgg_conv_block([in_channle * 2], [in_channle * 2], [3, 3], [1, 1], 2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # FC layers
        self.layer6 = vgg_fc_layer(4 * 4 * in_channle * 2, 2 * 2 * in_channle)
        self.layer7 = vgg_fc_layer(2 * 2 * in_channle, 512)

        # Final layer
        self.layer8 = tnn.Linear(512, num_components)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 提取特征
        vgg16_features = self.adaptive_pool(out)
        out = vgg16_features.view(out.size(0), -1)  # 展平特征
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out  # 返回特征和输出


if __name__ == '__main__':
    # 实例化模型并转移到 GPU
    vgg16 = VGG16(n_classes=768).cuda()

    # 输入数据
    input_tensor = torch.randn(1, 256, 64, 64).cuda()
    output = vgg16(input_tensor)

    print("Output Shape:", output.shape)  # 分类输出形状
    torch.save(vgg16.state_dict(), "vgg16.pth")
