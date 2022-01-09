import numpy as np
from tester import * 

def test_score(data, label, th, th0):
    return np.sum(label == np.sign(np.dot(th.T,data) + th0))

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    
    d, n = data.shape
    theta = np.zeros((d,1))
    theta_0 = np.zeros(1)
    #print("d = {}, n = {}, theta shape = {}, theta_0 shape = {}".format(d,n,theta.shape,theta_0.shape))
    #print(np.shape(theta),np.shape(theta_0))
    for i in range(T):
        for j in range(n):
            y = labels[0,j]
            x = data[:,j]
            a = np.dot(x,theta) +theta_0
            
            if np.sign(a*y)<=0:
                theta = theta + y*(x.reshape(d,1))
                theta_0 = theta_0 + y
    return theta, theta_0
    
def perceptron2(data, labels, params = {}, hook = None):    
    # if T not in params, default to 100
    T = params.get('T', 100)
    # Your implementation here
    d, n = data.shape
    theta = np.zeros((d,1))
    theta_0 = np.zeros(1)
    print("d = {}, n = {}, theta shape = {}, theta_0 shape = {}".format(d,n,theta.shape,theta_0.shape))
  
    for t in range(T):     
      for i in range(n):
        y = labels[0,i]
        x = data[:,i]
        
        a = np.dot(x,theta)+theta_0
        #print("a = {}".format(a))
        positive = np.sign(y*a)
        
        if np.sign(y*a) <=0: # update the thetas
          theta[:,0] = theta[:,0]+ y*x
          theta_0 = theta_0 + y
          
    print("shape x = {}, y = {}, theta = {}, theta_0 = {}".format(x.shape,y.shape,theta.shape,theta_0.shape))
    return (theta,theta_0)
    
test_perceptron(perceptron) 

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    #theta, theta_0 = learner(data_train, labels_train) #Without loss of generality
    theta, theta_0 = learner(data_train, labels_train)
    #print(theta,theta_0)
    return (test_score(data_test, labels_test, theta, theta_0))/data_test.shape[1]
    
test_eval_classifier(eval_classifier,perceptron)    

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    acc = 0
    for i in range(it):
        d_train, l_train = data_gen(n_train)
        d_test, l_test = data_gen(n_test)
        acc += eval_classifier(learner,d_train,l_train,d_test,l_test)
        
    return acc/it                                         

test_eval_learning_alg(eval_learning_alg,perceptron)

def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k
    
test_xval_learning_alg(xval_learning_alg,perceptron)
    
    
    
    
    
    

