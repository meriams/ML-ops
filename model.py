# Packages:
import torch.nn as nn 
import torch.nn.functional as F

class EmotionNet(nn.Module):
    network_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

    def __init__(self, numOfChannels, numOfClasses): 
        super(EmotionNet, self).__init__()
        self.features = self._make_layers(numOfChannels, self.network_config)
        self.classifier = nn.Sequential(nn.Linear(6 * 6 * 128, 64),
                                        nn.ELU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(64,numOfClasses))

    def forward(self, x):
        out = self.feautures(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=True)
        out = self.classifier(out)
        return out 

    # generate convolution layers in the model 
    def _make_layers(self, in_channel, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), 
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
                in_channel = x
        return nn.Sequential(*layers)





