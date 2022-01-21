import numpy as np
np.random.seed(1)
num_samples = 1001
noise = 0.1*np.random.standard_normal(num_samples)
X_left = np.random.uniform(low=0, high=1, size=(num_samples,6))
X_right = np.random.uniform(low=0.6, high=1, size=(num_samples,4))
x0,x1,x2,x5,x6,x8=X_left[:,0],X_left[:,1],X_left[:,2],X_left[:,3],X_left[:,4],X_left[:,5]
x3,x4,x7,x9=X_right[:,0],X_right[:,1],X_right[:,2],X_right[:,3]
y=np.pi**(x0*x1)*np.sqrt(2*x2)-np.arcsin(x3)+np.log(x2+x4)-x8/x9*np.sqrt(x6/x7)-x1*x6+noise
DATA=np.concatenate((x0.reshape(num_samples,1),
                     x1.reshape(num_samples,1),
                     x2.reshape(num_samples,1),
                     x3.reshape(num_samples,1),
                     x4.reshape(num_samples,1),
                     x5.reshape(num_samples,1),
                     x6.reshape(num_samples,1),
                     x7.reshape(num_samples,1),
                     x8.reshape(num_samples,1),
                     x9.reshape(num_samples,1),
                     y.reshape(num_samples,1)),axis=1)
np.savetxt('regression_data/'+str(0)+'.csv', DATA, delimiter=",")
