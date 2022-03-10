from torch import nn


class Classifier(nn.Module):
    def __init__(self, cfgs):
        super(Classifier, self).__init__()
        self.dims = cfgs['cls']['dims']
        self.prob = cfgs['cls']['prob']
        self.heads = self.get_heads(self.dims, self.prob)

    @staticmethod
    def get_heads(dims, prob):
        heads = []
        for i in range(1, len(dims) - 1):
            heads.append(nn.Linear(dims[i-1], dims[i]))
            heads.append(nn.Dropout(prob))

        heads.append(nn.Linear(dims[len(dims) - 2], dims[len(dims) - 1]))

        return nn.Sequential(*heads)

    def forward(self, input_tensor):
        output_tensor = self.heads(input_tensor)

        return output_tensor
