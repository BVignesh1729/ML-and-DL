#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing Pipeline

# In[1]:


import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ### Tokenization and Stemming

# In[2]:


nltk.download('punkt')
def tokenize(sentence): 
    return nltk.word_tokenize(sentence)


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stemming(word):
    return stemmer.stem(word.lower())

    


# In[3]:


s = "How long does shipping take?"
print(s)
print(tokenize(s))


# In[4]:


similar1 = ['happy', 'happier', 'happiest']
similar2 = ['organize', 'organizer', 'organizing']

stemmed_words1 = [stemming(w) for w in similar1]
stemmed_words2 = [stemming(w) for w in similar2]


# In[5]:


stemmed_words1, stemmed_words2


# In[6]:


import json

with open('data.json','r') as f:
    intents = json.load(f)
    
print(intents)


# ### Collecting all the words

# In[7]:


all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']: 
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
        
# print(all_words)
ignore_words = ['?', '!', '.', ',']
all_words = [stemming(w) for w in all_words if w not in ignore_words]

all_words =sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)
    


# #### Defining the bag of words function, to vectorize the tokenized sentences

# In[8]:


def bag_of_words(sentence,bag):
    sentence = [stemming(w) for w in sentence]
    vec = np.zeros(len(bag),dtype=np.float32)
    for i,word in enumerate(bag):
        if word not in sentence:
            vec[i] = 0.0
        else:
            vec[i] = 1.0
                
    return vec


# In[9]:


sentence = ["hello", "how", "are", "you", "thanks"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

k = bag_of_words(sentence,words)
k


# #### Vectorizing the data and forming the training set array

# In[10]:


X_train = []
y_train = [] 
for (pattern_sentence, tag) in  xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) 
    
X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape,y_train.shape)


# ## Creating our dataset, model and defining the hyper parameters

# In[11]:


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
        


# ### Defining the neural net

# In[12]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1=nn.Linear(input_size, hidden_size)
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.l3=nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No softmax for now, cross entrpy loss applies it by itself
        
        return out
        
        


# ### Defining the hyperparameters

# In[13]:


#Hyper Parameters

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)


# ### Defining the loss and optimizer function

# In[14]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[15]:


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        #Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1)%100==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')
print(f'Final loss is {loss.item():.4f}')

    


# In[16]:


data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags,
    
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training complete, file saved to {FILE}")


# In[ ]:




