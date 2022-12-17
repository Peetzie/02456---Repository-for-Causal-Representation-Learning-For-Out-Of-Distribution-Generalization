#classifier NN
from sklearn import metrics
from typing import *
from torch import nn

#Accuracy function from week 4:
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

#Building a classifier:

class Classifier(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = len(indices)
        #Layers
        self.input_layer = nn.Linear(in_features=self.indices, out_features = 50)
        self.layer1 = nn.Linear(in_features = 50, out_features = 100)
        self.output_layer = nn.Linear(in_features = 100, out_features = 1)

        #activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    #forward function(self, x):
    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.output_layer(x))

        return x
    # 
