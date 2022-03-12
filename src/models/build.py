from torch import nn
import torch
from torchsummary import summary
import  math
import sys
sys.path.append('./models')
sys.path.append('./models/extractors')
sys.path.append('../src')
from convnext import build_extractor
from head import Classifier
from utils import get_config
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    def __init__(self, extractor, head):
        super(Model, self).__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, input_tensor):
        features = self.extractor(input_tensor)
        categorical_probs = self.head(features)

        return features, categorical_probs


def build_model(cfgs, pretrained=False, **kwargs):
    extractor = build_extractor(cfgs, pretrained, **kwargs)
    head = Classifier(cfgs)

    return Model(extractor, head)


def effi():
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=9)
    return model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class SPP(nn.Module):
    def __init__(self, out_pool_size):
        super(SPP, self).__init__()
        self.out_pool_size = out_pool_size

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        for i in range(len(self.out_pool_size)):
            h_wid = int(math.ceil(height / self.out_pool_size[i]))
            w_wid = int(math.ceil(width / self.out_pool_size[i]))
            h_pad = int((h_wid * self.out_pool_size[i] - height + 1) / 2)
            w_pad = int((w_wid * self.out_pool_size[i] - width + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            out = maxpool(x)
            if i == 0:
                spp = out.view(batch_size, -1)
            else:
                spp = torch.cat((spp, out.view(batch_size, -1)), 1)
        return spp


class EfficientNetSpp(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNetSpp, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0', include_top=False)
        self.base._fc = Identity()
        for name, layer in self.base.named_children():
            if isinstance(layer, nn.Linear):
                self.base.add_module(name, Identity())
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                self.base.add_module(name, Identity())
        
        self.proj = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(), 
            nn.BatchNorm2d(256)
        )
        self.spp = SPP([1, 2, 3, 4])
        self.fc = nn.Linear(7680, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.proj(x)
        feat = self.spp(x)
        out = self.fc(feat)
        return out

if __name__ == '__main__':
    cfgs = get_config('../config.yml')
    model = build_model(cfgs)
    summary(model, input_size=(3, 64, 64), batch_size=2)
    pass
