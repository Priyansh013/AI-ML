import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse



parser = argparse.ArgumentParser(description="Pass parameters for this Model: seed , Learning Rate")

parser.add_argument("--seed", metavar="Seed",type=int, help="Seed Value")
parser.add_argument("--epochs", metavar="Seed",type=int, help="Number of Epochs")
parser.add_argument("--alpha",metavar="Learning Rate",type=float,help="Learning Rate enter")


args = parser.parse_args()



seed = args.seed
alpha = args.alpha
epochs = args.epochs




df = pd.read_csv("diabetes.csv")
df.head()

X  = df.drop('Outcome',axis=1).values   #.values to convert pandas df to numpy array
y = df['Outcome'].values


X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)




# COnverting into Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)





# Creating Model


class ANN_Model(nn.Module):
    def __init__(self,input_feature=8,h1=20,h2=20,output_feature=2):
        super().__init__()
        self.fc1 = nn.Linear(input_feature,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,output_feature)


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x 
    
    
    
    
    
# Instantiate model:
torch.manual_seed(seed)
model = ANN_Model()



# Defining loss and optimisers:

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=alpha)



# Running Model:


losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred,y_train)
    losses.append(loss.detach().numpy())
    
    if i%50==0:
        print(f"Epoch: {i} , Loss: {loss.item()}")
    
    
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
        

