import numpy as np

#Theta 0 is a scalar for a single hypothesis
#Say we work with R2

#For a single hypothesis single data point
theta = np.array([[1,-2]]).T
theta_0 = 2
point = np.array([[-2,1]]).T

def classify(p,th,th_0):
    return np.sign(np.dot(th.T,p)+th_0)
print(classify(point,theta,theta_0))

#Note : For a single hypothesis and a single point, theta_0 can remain a scalar

#For multiple data points

data = np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]).T
labels = np.array([[-1, -1, +1, +1, +1]])
def mul_classify(d,th,th_0):
    return np.sum(labels == np.sign(np.dot(th.T,d)+th_0))

print(mul_classify(data,theta,theta_0))

#For multiple hypothesis

mul_th = np.array([[1,-2],[-1,2],[0,4]]).T
mul_th_0 = np.array([[2,3,-1]])

def mclass_mhp(d,mth,mth_0):
    return np.sum((np.concatenate((labels,labels,labels),axis=0) == np.sign(np.dot(mth.T,d)+mth_0.T)),axis=1)

def find_hp(d,mth,mth_0):
    idx = mclass_mhp(d,mth,mth_0)
    return np.argmax(idx)

print(mclass_mhp(data,mul_th,mul_th_0))

print('For the best hypothesis, theta: ',mul_th[:,find_hp(data,mul_th,mul_th_0)],'with offest theta_0: ',mul_th_0[:,find_hp(data,mul_th,mul_th_0)])

