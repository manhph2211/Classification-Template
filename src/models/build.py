from torch import nn
from torchsummary import summary

import sys
sys.path.append('./models')
sys.path.append('./models/extractors')
sys.path.append('../src')
from convnext import build_extractor
from head import Classifier
from utils import get_config


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


if __name__ == '__main__':
    cfgs = get_config('../config.yml')
    model = build_model(cfgs)
    summary(model, input_size=(3, 64, 64), batch_size=2)
    pass
