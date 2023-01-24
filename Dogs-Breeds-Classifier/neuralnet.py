import torch.nn.functional as F
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(1024, 500)
        self.fc_2 = nn.Linear(500, 300)
        self.output = nn.Linear(300, 120)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.dropout(F.relu(self.fc_2(x)))
        x = F.log_softmax(self.output(x), dim=1)

        return x