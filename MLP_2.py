
# coding: utf-8

# # MLP

# ## Imports

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import sys


# ## Data to loaders

# In[2]:



# how many samples per batch to load
batch_size = 3
# percentage of training set to use as validation
valid_size = 0.2


# choose the training and test datasets
data = pd.read_csv('test_data.csv')
train_data = data.drop('Unnamed: 0', axis=1).drop('casual', axis=1).drop('registered', axis=1)

# Features and targets
features_train = data.drop('cnt', axis=1)
targets_train = data['cnt']

# To tensors
features_tensor = torch.tensor(features_train.values, dtype=torch.float, requires_grad=False)
targets_tensor = torch.tensor(targets_train.values, dtype=torch.float)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# To dataset
trainDataSet = torch.utils.data.TensorDataset(features_tensor, targets_tensor)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=batch_size,
    sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=batch_size, 
    sampler=valid_sampler)


# In[3]:


print(features_train.shape[1])


# ## Define network

# In[4]:


# define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_1 = 16
        hidden_2 = 8
        
        self.fc1 = nn.Linear(features_train.shape[1], hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)


    def forward(self, x):
        
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.out(x)
        
        return x

model = Net()
print(model)


# ## Loss and optimizer

# In[5]:


# specify loss function
criterion = nn.MSELoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.2)


# ## Train

# In[6]:


# number of epochs to train the model
n_epochs = 50

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0
    
    for data, target in train_loader:
       
        optimizer.zero_grad()
    
        output = model(data)
        
        #output = output.squeeze(1)
        
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
       
        train_loss += loss.item()*data.size(0)
  
    for data, target in valid_loader:
       
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
    


# ## Load model

# In[ ]:


model.load_state_dict(torch.load('model_cifar.pt'))


# ## Test model

# In[ ]:


# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

