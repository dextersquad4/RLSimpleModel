import torch.nn as nn
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # The two features are:
            # The car's distance from the left which ranges from 0-10 float (if it reaches 0 or 10 the reward sequence stops)
            # The angle with ranges from -1 to 1 (which signify -90 and 90 degrees)
        self.linear1 = nn.Linear(2, 128) 
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        outputs = self.linear3(x)
        mean_raw, std_raw = outputs[0], outputs[1]
        return torch.tanh(mean_raw), std_raw
    
