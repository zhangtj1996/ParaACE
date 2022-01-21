import numpy as np
import os
directorys=['regression_data','classification_data']  # temp models
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
num_samples = 12501
def synth_func(X):
    funcList=[]
    #F2
    funcList.append(
    np.pi**(X[:,0]*X[:,1])*np.sqrt(2*np.abs(X[:,2]))-np.arcsin(X[:,3]/2)+np.log(np.abs(X[:,4]+X[:,2])+1)-(X[:,8]/1+np.abs(X[:,9]))*np.sqrt(np.abs(X[:,6])/(1+np.abs(X[:,7])))-X[:,1]*X[:,6]
    )    
    #F3
    funcList.append(
        np.exp(np.abs(X[:,0]-X[:,1]))+np.abs(X[:,2]*X[:,1])-(X[:,2]**2)*np.abs(X[:,3])+np.log(X[:,3]**2+X[:,4]**2+X[:,6]**2+X[:,7]**2 )+X[:,8]+1/(1+X[:,9]**2) 
    )
    #F4
    funcList.append(
        np.exp(np.abs(X[:,0]-X[:,1]))+np.abs(X[:,2]*X[:,1])-(X[:,2]**2)*np.abs(X[:,3])+(X[:,0]*X[:,3])**2+np.log(X[:,3]**2+X[:,4]**2+X[:,6]**2+X[:,7]**2 )+X[:,8]+1/(1+X[:,9]**2) 
    )  
    #F5
    funcList.append(
    1/(1+X[:,0]**2+X[:,1]**2+X[:,2]**2)+np.sqrt(np.exp(X[:,3]+X[:,4]))+np.abs(X[:,5]+X[:,6])+X[:,7]*X[:,8]*X[:,9]
    )    
    #F6
    funcList.append(
        np.exp(np.abs(X[:,0]*X[:,1])+1)-np.exp(np.abs(X[:,2]+X[:,3])+1)+np.cos(X[:,4]+X[:,5]-X[:,7])+np.sqrt(X[:,7]**2+X[:,8]**2+X[:,9]**2) 
    )      
    #F7
    funcList.append(
        (np.arctan(X[:,0])+np.arctan(X[:,1]))**2+np.maximum(X[:,2]*X[:,3]+X[:,5],0)-1/(1+(X[:,3]*X[:,4]*X[:,5]*X[:,6]*X[:,7])**2)+(np.abs(X[:,6])/(1+np.abs(X[:,8])))**5+
        X[:,0]+X[:,1]+X[:,2]+X[:,3]+X[:,4]+X[:,5]+X[:,6]+X[:,7]+X[:,8]+X[:,9]
    )
    #F8
    funcList.append(
    X[:,0]*X[:,1]+2**(X[:,2]+X[:,4]+X[:,5])+2**(X[:,2]+X[:,3]+X[:,4]+X[:,6])+np.sin(X[:,6]*np.sin(X[:,7]+X[:,8]))+np.arccos(0.9*X[:,9])
    )    
    #F9
    funcList.append(
    np.tanh(X[:,0]*X[:,1]+X[:,2]*X[:,3])*np.sqrt(np.abs(X[:,4]))+np.exp(X[:,4]+X[:,5])+np.log((X[:,5]*X[:,6]*X[:,7])**2+1)+X[:,8]*X[:,9]+1/(1+np.abs(X[:,9]))
    )
    #F10
    funcList.append(
    np.sinh(X[:,0]+X[:,1])+np.arccos(np.tanh(X[:,2]+X[:,4]+X[:,6]))+np.cos(X[:,3]+X[:,4])+ 1/np.cos(X[:,6]*X[:,8])
    )

    return funcList

def gen_synth_data(k,classification=True):
    np.random.seed(0)
    X = np.random.uniform(low=-1, high=1, size=(num_samples,10))
    noise = 0.1*np.random.standard_normal(num_samples)
    if classification:
        Y = np.expand_dims(synth_func(X)[k],axis=1)# +noise,axis=1)
        cc=Y>np.median(Y)
        dd=Y<np.median(Y)
        Y[cc]=1
        Y[dd]=0
        DATA=np.concatenate((X,Y),axis=1)
        return DATA
    else:
        Y = np.expand_dims(synth_func(X)[k],axis=1)
        DATA=np.concatenate((X,Y),axis=1)
        return DATA

    
for i in range(9):
    print(i)
    functioni=i
    DATA = gen_synth_data(functioni,classification=False)
    np.savetxt('regression_data/'+str(i+1)+'.csv', DATA, delimiter=",")
    
for i in range(9):
    print(i)    
    functioni=i
    DATA = gen_synth_data(functioni,classification=True)
    np.savetxt('classification_data/'+str(i+1)+'.csv', DATA, delimiter=",")