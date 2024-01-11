import torch.nn as nn
import torch

# the model is based on the work 'Facial Expression Recognition based on Convolutional Denoising Autoencoder and XGBoost

class CAE_finger(nn.Module):
    # this model involves max pooling on encoder and upsample layer on decoder
    # the same model with the reference work
    def __init__(self, num_layers, in_channels, disable_decoder=False, bias=False):
        super(CAE_finger, self).__init__()
        self.disable_decoder = disable_decoder
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        self.channel_step = 5
        self.bias = bias

        self.encoder = self.make_encoder(self.num_layers)
        self.fcE = nn.Linear(in_features=1024*1*2, out_features=512)
        self.fcD = nn.Linear(in_features=512, out_features=1024*1*2)
        self.decoder = self.make_decoder(self.num_layers)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1, stride=1)

        if self.bias:
            self.output_bias = nn.Parameter(torch.zeros(1, 128, 256), requires_grad=True)

        self.apply(self.init_weights)

    def forward(self, input):
        # encode
        out = self.encoder(input)
        out = out.view(-1, 1024*1*2)
        out = self.fcE(out)
        latent = out
        out = self.fcD(out)
        out = out.view(-1, 1024, 1, 2)
        out = self.decoder(out)
        out = self.conv(out)
        if self.bias:
            out = out + self.output_bias
        if self.disable_decoder:
            return out, latent

        return out

    def make_encoder(self, num_layers):
        encoder_layers = []
        out_channels = self.in_channels
        for l in range(num_layers):
            if l == 0:
                encoder_layers.append(self.conv_block(in_channels=1, out_channels=self.in_channels,
                                                  kernel_size=3, stride=2, padding=1))
                in_channels = self.in_channels
            else:
                out_channels = in_channels * 2
                encoder_layers.append(self.conv_block(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=3, stride=2, padding=1))
                in_channels = out_channels
            self.out_channels = out_channels
        return nn.Sequential(*encoder_layers)

    def make_decoder(self, num_layers):
        decoder_layers = []
        in_channels = self.out_channels
        for l in range(num_layers):
            padding = 0
            if l == num_layers - 1:
                out_channels = in_channels
            else:
                out_channels = in_channels // 2
            decoder_layers.append(self.deconv_block(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=2, stride=2, padding=padding))
            in_channels = out_channels

        return nn.Sequential(*decoder_layers)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        return block

    def deconv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        return block

    def init_weights(self, x):
        if isinstance(x, nn.Conv2d):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
        if isinstance(x, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
        if isinstance(x, nn.Linear):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
        if isinstance(x, nn.BatchNorm2d):
            nn.init.normal_(x.weight.data, 1.0, 0.02)
            nn.init.constant_(x.bias.data, 0)
