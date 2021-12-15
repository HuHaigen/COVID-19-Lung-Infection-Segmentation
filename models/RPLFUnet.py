import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_relu = with_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x, x1, x2, x3, x4]


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(
            h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.bridge(x)
        return x


class after_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bridge(x)
        x = F.interpolate(x, (256, 256), mode='bilinear')
        forg = self.conv(x)
        g = self.sigmoid(forg)
        return x, g


class up_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bridge(x)
        x = self.sigmoid(x)
        return x


class RPLFUnet(nn.Module):
    def __init__(self, n_classes=1, **kwargs):
        super().__init__()
        self.ResNet = ResNet(ResBlock, [3, 4, 6, 3])
        # self.w=nn.ParameterList([nn.Parameter(torch.ones(1),requires_grad=True)for i in range(5)])
        self.bridge = Bridge(512, 512)
        # self.psp = PSPModule(512, 512, (1, 2, 3, 6))
        up_blocks = []
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet50(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet50(96, 64, 64, 32))
        self.final_upsample = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out1 = nn.Conv2d(64, 1, 1, 1, bias=False)

        self.out = []
        self.out_ = res(64, 1).cuda()
        self.out.append(res(64, 1).cuda())
        self.out.append(res(64, 1).cuda())
        self.out.append(res(128, 1).cuda())
        self.out.append(res(256, 1).cuda())
        self.out.append(res(512, 1).cuda())

        self.fout = []
        self.fout.append(after_res(64, 64).cuda())
        self.fout.append(after_res(64, 64).cuda())
        self.fout.append(after_res(128, 64).cuda())
        self.fout.append(after_res(256, 64).cuda())
        self.fout.append(after_res(512, 64).cuda())
        self.up = up_res(64, 1).cuda()
        self.gate = after_res(64, 64).cuda()

    def forward(self, x, with_output_feature_map=False):
        down_x = self.ResNet(x)
        put_x = []
        # for i in range(5):
        #     put_x.append(self.out[i](down_x[i]))

        x = self.bridge(down_x[4])

        after_x = []
        for i, block in enumerate(self.up_blocks, 1):
            after_x.append(x)
            x = block(x, down_x[4 - i])
        after_x.append(x)

        x = self.final_upsample(x)
        x = self.final_conv(x)

        gs = []
        xs = []
        xs_256 = []
        for i in range(len(after_x)):
            xx, g = self.fout[i](after_x[4 - i])
            gs.append(g)
            xs.append(xx)
            xs_256.append(F.sigmoid(self.up(xx)))
        other = gs[0] * xs[0]
        for i in range(1, len(gs)):
            other += gs[i] * xs[i]

        x, gx = self.gate(x)
        others = other * (1 - gx)
        x1s = gx * x
        x = others + x1s + x

        x = self.out1(x)
        return put_x, F.sigmoid(x), xs_256
