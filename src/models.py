import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBNReLU(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBNReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layer(x)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_dconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_dconv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_encoder, in_channels_decoder, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
            self.conv1 = ConvBNReLU(in_channels_encoder + in_channels_decoder, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_decoder, in_channels_encoder, kernel_size=3, stride=2)
            
            self.conv1 = ConvBNReLU(in_channels_encoder + in_channels_encoder, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet_Loc(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Unet_Loc, self).__init__()

        encoder_filters = [64, 128, 256, 512, 1024]
        decoder_filters = [1024, 512, 256, 128, 64]

        self.inc = DoubleConv(n_channels, encoder_filters[0])

        self.down1 = Down(encoder_filters[0], encoder_filters[1])
        self.down2 = Down(encoder_filters[1], encoder_filters[2])
        self.down3 = Down(encoder_filters[2], encoder_filters[3])
        self.down4 = Down(encoder_filters[3], encoder_filters[4])
        
        self.up1 = Up(encoder_filters[3], decoder_filters[0], decoder_filters[1], bilinear)
        self.up2 = Up(encoder_filters[2], decoder_filters[1], decoder_filters[2], bilinear)
        self.up3 = Up(encoder_filters[1], decoder_filters[2], decoder_filters[3], bilinear)
        self.up4 = Up(encoder_filters[0], decoder_filters[3], decoder_filters[4], bilinear)

        self.outc = OutConv(decoder_filters[4], n_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.inc(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
                        
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)
        
        out = self.outc(up4)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
class UNet_Double(nn.Module):

    def __init__(self, n_channels=3, n_classes=5, bilinear=True):
        super(UNet_Double, self).__init__()
                
        encoder_filters = [64, 128, 256, 512, 1024]
        decoder_filters = [1024, 512, 256, 128, 64]

        self.inc = DoubleConv(n_channels, encoder_filters[0])

        self.down1 = Down(encoder_filters[0], encoder_filters[1])
        self.down2 = Down(encoder_filters[1], encoder_filters[2])
        self.down3 = Down(encoder_filters[2], encoder_filters[3])
        self.down4 = Down(encoder_filters[3], encoder_filters[4])
        
        self.up1 = Up(encoder_filters[3], decoder_filters[0], decoder_filters[1], bilinear)
        self.up2 = Up(encoder_filters[2], decoder_filters[1], decoder_filters[2], bilinear)
        self.up3 = Up(encoder_filters[1], decoder_filters[2], decoder_filters[3], bilinear)
        self.up4 = Up(encoder_filters[0], decoder_filters[3], decoder_filters[4], bilinear)

        self.outc = OutConv(decoder_filters[4]*2, n_classes)

        self._initialize_weights()
        
    def forward_1(self, x):
        x = self.inc(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
                        
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)
        
        return up4
    
    def forward(self, x):        
        x1 = self.forward_1(x[:, :3, :, :])
        x2 = self.forward_1(x[:, 3:, :, :])
        x = torch.cat([x1, x2], 1)
        out = self.outc(x)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    

class ResNet34_Unet_Loc(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, pretrained=False):
        super(ResNet34_Unet_Loc, self).__init__()

        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = [512, 256, 128, 64, 64]
        
        self.inc = DoubleConv(n_channels, encoder_filters[0])
        
        self.up1 = Up(encoder_filters[3], decoder_filters[0], decoder_filters[1], bilinear)
        self.up2 = Up(encoder_filters[2], decoder_filters[1], decoder_filters[2], bilinear)
        self.up3 = Up(encoder_filters[1], decoder_filters[2], decoder_filters[3], bilinear)
        self.up4 = Up(encoder_filters[0], decoder_filters[3], decoder_filters[4], bilinear)

        self.outc = OutConv(decoder_filters[4], n_classes)
        self._initialize_weights()
        
        encoder = torchvision.models.resnet34(pretrained=pretrained)
        
        self.down1 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.down2 = encoder.layer2
        self.down3 = encoder.layer3
        self.down4 = encoder.layer4

    def forward(self, x):
        x = self.inc(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
                        
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)
        
        out = self.outc(up4)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet34_Unet_Double(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, bilinear=True, pretrained=False):
        super(ResNet34_Unet_Double, self).__init__()

        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = [512, 256, 128, 64, 64]

        self.inc = DoubleConv(n_channels, encoder_filters[0])
        
        self.up1 = Up(encoder_filters[3], decoder_filters[0], decoder_filters[1], bilinear)
        self.up2 = Up(encoder_filters[2], decoder_filters[1], decoder_filters[2], bilinear)
        self.up3 = Up(encoder_filters[1], decoder_filters[2], decoder_filters[3], bilinear)
        self.up4 = Up(encoder_filters[0], decoder_filters[3], decoder_filters[4], bilinear)

        self.outc = OutConv(decoder_filters[4] * 2, n_classes)
        self._initialize_weights()

        encoder = torchvision.models.resnet34(pretrained=pretrained)
        
        self.down1 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.down2 = encoder.layer2
        self.down3 = encoder.layer3
        self.down4 = encoder.layer4

    def forward_1(self, x):
        x = self.inc(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
                        
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)
        
        return up4
    
    def forward(self, x):        
        x1 = self.forward_1(x[:, :3, :, :])
        x2 = self.forward_1(x[:, 3:, :, :])
        x = torch.cat([x1, x2], 1)
        out = self.outc(x)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet50_Unet_Loc(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, pretrained=False):
        super(ResNet50_Unet_Loc, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = [2048, 1024, 512, 256, 64]

        self.inc = DoubleConv(n_channels, encoder_filters[0])
        
        self.up1 = Up(encoder_filters[3], decoder_filters[0], decoder_filters[1], bilinear)
        self.up2 = Up(encoder_filters[2], decoder_filters[1], decoder_filters[2], bilinear)
        self.up3 = Up(encoder_filters[1], decoder_filters[2], decoder_filters[3], bilinear)
        self.up4 = Up(encoder_filters[0], decoder_filters[3], decoder_filters[4], bilinear)

        self.outc = OutConv(decoder_filters[4], n_classes)
        self._initialize_weights()

        encoder = torchvision.models.resnet50(pretrained=pretrained)
        
        self.down1 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.down2 = encoder.layer2
        self.down3 = encoder.layer3
        self.down4 = encoder.layer4

    def forward(self, x):
        x = self.inc(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
                        
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)
        
        out = self.outc(up4)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet50_Unet_Double(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, bilinear=True, pretrained=False):
        super(ResNet50_Unet_Double, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = [2048, 1024, 512, 256, 64]

        self.inc = DoubleConv(n_channels, encoder_filters[0])
        
        self.up1 = Up(encoder_filters[3], decoder_filters[0], decoder_filters[1], bilinear)
        self.up2 = Up(encoder_filters[2], decoder_filters[1], decoder_filters[2], bilinear)
        self.up3 = Up(encoder_filters[1], decoder_filters[2], decoder_filters[3], bilinear)
        self.up4 = Up(encoder_filters[0], decoder_filters[3], decoder_filters[4], bilinear)

        self.outc = OutConv(decoder_filters[4] * 2, n_classes)
        self._initialize_weights()
         
        encoder = torchvision.models.resnet50(pretrained=pretrained)
        
        self.down1 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.down2 = encoder.layer2
        self.down3 = encoder.layer3
        self.down4 = encoder.layer4

    def forward_1(self, x):
        x = self.inc(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
                        
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)
        
        return up4
    
    def forward(self, x):        
        x1 = self.forward_1(x[:, :3, :, :])
        x2 = self.forward_1(x[:, 3:, :, :])
        x = torch.cat([x1, x2], 1)
        out = self.outc(x)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
