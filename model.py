import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        __init__().super()
        # The two features are:
            # The car's distance from the left which ranges from 0-10 float (if it reaches 0 or 10 the reward sequence stops)
            # The angle with ranges from -90 to 90 degrees
        self.linear1 = nn.Linear(2, 2) 
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
    
