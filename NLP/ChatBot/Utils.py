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


def bag_of_words(sentence,bag):
    sentence = [stemming(w) for w in sentence]
    vec = np.zeros(len(bag),dtype=np.float32)
    for i,word in enumerate(bag):
        if word not in sentence:
            vec[i] = 0.0
        else:
            vec[i] = 1.0
                
    return vec

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