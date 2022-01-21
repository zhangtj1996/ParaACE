import numpy as np
import time
import torch
from torch.autograd import Variable
import itertools, pickle

def one_hot(i,p):
    batch_size=1
    # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    y = torch.LongTensor([[i]])
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, p)
    #print(y_onehot)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot
def detect_Hessian_UCB(FCnet,X_train,K):
    np.random.seed(0)
    def evaluate_2nd_derivative(net,i,k,N):
        # four terms need to be evaluated
        delta=1.5   #np.random.rand()+0.5
        randi=np.random.randint(N)
        xeval = Variable(X_train[randi,:], requires_grad=True)#pick one sample randomly
        xeval=xeval.reshape(1,p) 
        #double side
        f0 = net(xeval-one_hot(i,p)*delta-one_hot(k,p)*delta)
        fi =net(xeval+one_hot(i,p)*delta-one_hot(k,p)*delta)
        fik =net(xeval+one_hot(i,p)*delta+one_hot(k,p)*delta)
        fk=net(xeval+one_hot(k,p)*delta-one_hot(i,p)*delta)
        
        inter_strength=(fik-fi-fk+f0)/(delta**2)
        reward=inter_strength.detach().numpy()**2 #abs()
        return -float(reward)

    print("start dectecting")    
    st=time.time()
    N=X_train.shape[0]
    p=X_train.shape[1]
    Larms=[]
    for i in itertools.combinations(range(p),2):
        Larms.append(i)
    
    n=len(Larms)
    Delta = 1.0/n 
    step_size=1
    num_arms=1
    lcb = np.zeros(n, dtype='float')       #At any point, stores the mu - lower_confidence_interval
    ucb = np.zeros(n, dtype='float')       #At any point, stores the mu + lower_confidence_interval
    T = step_size*np.ones(n, dtype='int')#At any point, stores number of times each arm is pulled

    n_init_try=3
    init_data=np.zeros((n_init_try,n))
    for j in range(n_init_try):
        for i in range(n):
            init_data[j,i]=evaluate_2nd_derivative(FCnet,Larms[i][0],Larms[i][1],N)
            
    maxrecord=np.max(init_data,axis=0)
    minrecord=np.min(init_data,axis=0)
    sigma=(maxrecord-minrecord)/2
            
            
    estimate=np.mean(init_data,axis=0)
    lcb = estimate - np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    ucb = estimate + np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    print("Initialization done! initial try"+str(n_init_try)+'times')
    #print(estimate,sigma,np.sqrt(sigma**2*np.log(1/Delta)/step_size))


    def choose_arm():
        n=100 # MAXPULL
        low_lcb_arms = np.argpartition(lcb,num_arms)[:num_arms]
        arms_pulled_morethan_n = low_lcb_arms[ np.where( (T[low_lcb_arms]>=n) & (ucb[low_lcb_arms] != lcb[low_lcb_arms]) ) ]
        if arms_pulled_morethan_n.shape[0]>0:
            # Compute the distance of these arms accurately
            #print('more_than_n')
            #estimate[arms_pulled_morethan_n] = evaluate_2nd_derivative_brute(net,Larms[int(arms_pulled_morethan_n)][0],Larms[int(arms_pulled_morethan_n)][1])
            #T[arms_pulled_morethan_n]  += n
            ucb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            lcb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            return None
        
        if ucb.min() <  lcb[np.argpartition(lcb,1)[1]]: #Exit condition
            return None
        arms_to_pull          = low_lcb_arms[ np.where(T[low_lcb_arms]<n) ]
        return arms_to_pull
    def pull_arm(arms,N):
        Tmean = evaluate_2nd_derivative(FCnet,Larms[int(arms)][0],Larms[int(arms)][1],N)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*step_size)/( T[arms] + step_size + 0.0 )
        T[arms]          = T[arms]+step_size
        lcb[arms]        = estimate[arms] - np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        ucb[arms]        = estimate[arms] + np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        maxrecord[arms]   = max(maxrecord[arms],estimate[arms])
        minrecord[arms]   = min(minrecord[arms],estimate[arms])
        sigma[arms]       = (maxrecord[arms]-minrecord[arms])/2 
        
    
    chosen_arms=[]
    record_chosen_arms=[]
    k,K=1,K #set pick interactions       
    cnt=0
    
    while k <=K :
        arms_to_pull=choose_arm()
        while arms_to_pull==None:
            
            chosen_arm_pos=np.argmin(ucb)
            chosen_arms.append(chosen_arm_pos) # add the chosen arm to K best arms
            # record the(position, ucb, mean, lcb,T)
            record_chosen_arms.append((chosen_arm_pos,
                                       ucb[chosen_arm_pos],
                                       estimate[chosen_arm_pos],
                                      lcb[chosen_arm_pos],
                                       T[chosen_arm_pos]
                                      ))
            print('chosen arm:',chosen_arm_pos,'strength:',-estimate[chosen_arm_pos], 'iteration:',cnt)
            # set the ucb mean lcb to be a large number, so this arm won't be pulled
            ucb[chosen_arm_pos],estimate[chosen_arm_pos],lcb[chosen_arm_pos]=0.5,0.5,0.5
            arms_to_pull=choose_arm()
            
            k=k+1    
        #print(arms_to_pull)
        pull_arm(arms_to_pull,N)
        cnt=cnt+1
    totaltime=time.time()-st
    print('time:',totaltime)
        
    #reset values    
    for i in record_chosen_arms:
        ucb[i[0]]=i[1]
        estimate[i[0]]=i[2]
        lcb[i[0]]=i[3]
    index_Subset=[]
    for i in chosen_arms:
        index_Subset.append(np.array(Larms[i])) #selected
         
    return index_Subset,totaltime


def detect_3rd_UCB(FCnet,X_train,Larms,K):
    np.random.seed(0)
    def evaluate_3rd_derivative(net,i,j,k,N):
        # eight terms need to be evaluated
        delta=1.5   #np.random.rand()+0.5
        randi=np.random.randint(N)
        xeval = Variable(X_train[randi,:], requires_grad=True)#pick one sample randomly
        xeval=xeval.reshape(1,p) 
        #double side
        f0 = net(xeval-one_hot(i,p)*delta+one_hot(j,p)*delta+one_hot(k,p)*delta)
        f1 = net(xeval+one_hot(i,p)*delta-one_hot(j,p)*delta+one_hot(k,p)*delta)
        f2 = net(xeval+one_hot(i,p)*delta+one_hot(j,p)*delta-one_hot(k,p)*delta)
        f3 = net(xeval-one_hot(i,p)*delta-one_hot(j,p)*delta-one_hot(k,p)*delta)
        
        f4 = net(xeval+one_hot(i,p)*delta+one_hot(j,p)*delta+one_hot(k,p)*delta)
        f5 = net(xeval-one_hot(i,p)*delta-one_hot(j,p)*delta+one_hot(k,p)*delta)
        f6 = net(xeval-one_hot(i,p)*delta+one_hot(j,p)*delta-one_hot(k,p)*delta)
        f7 = net(xeval+one_hot(i,p)*delta-one_hot(j,p)*delta-one_hot(k,p)*delta)
        
        inter_strength=(f4+f5+f6+f7-f0-f1-f2-f3)/(8*delta**3)
        reward=inter_strength.detach().numpy()**2 #abs()
        return -float(reward)
    print("start dectecting")    
    st=time.time()
    N=X_train.shape[0]
    p=X_train.shape[1]
  
    n=len(Larms)
    Delta = 1.0/n 
    step_size=1
    num_arms=1    #number of arms pulled in one iteration                    
    lcb = np.zeros(n, dtype='float')       #At any point, stores the mu - lower_confidence_interval
    ucb = np.zeros(n, dtype='float')       #At any point, stores the mu + lower_confidence_interval
    T = step_size*np.ones(n, dtype='int')#At any point, stores number of times each arm is pulled

    # initially pull each arm 3 times
    n_init_try=3
    init_data=np.zeros((n_init_try,n))
    for j in range(n_init_try):
        for i in range(n):
            init_data[j,i]=evaluate_3rd_derivative(FCnet,Larms[i][0],Larms[i][1],Larms[i][2],N)
            
    maxrecord=np.max(init_data,axis=0)
    minrecord=np.min(init_data,axis=0)
    sigma=(maxrecord-minrecord)/2
            
            
    estimate=np.mean(init_data,axis=0)
    lcb = estimate - np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    ucb = estimate + np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    print("Initialization done! initial try"+str(n_init_try)+'times')
    #print(estimate,sigma,np.sqrt(sigma**2*np.log(1/Delta)/step_size))


    def choose_arm():
        n=100 # MAXPULL
        low_lcb_arms = np.argpartition(lcb,num_arms)[:num_arms]
        arms_pulled_morethan_n = low_lcb_arms[ np.where( (T[low_lcb_arms]>=n) & (ucb[low_lcb_arms] != lcb[low_lcb_arms]) ) ]
        if arms_pulled_morethan_n.shape[0]>0:
            # Compute the distance of these arms accurately
            #print('more_than_n')
            #estimate[arms_pulled_morethan_n] = evaluate_2nd_derivative_brute(net,Larms[int(arms_pulled_morethan_n)][0],Larms[int(arms_pulled_morethan_n)][1])
            #T[arms_pulled_morethan_n]  += n
            ucb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            lcb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            return None
        
        if ucb.min() <  lcb[np.argpartition(lcb,1)[1]]: #Exit condition
            return None
        arms_to_pull          = low_lcb_arms[ np.where(T[low_lcb_arms]<n) ]
        return arms_to_pull
    def pull_arm(arms,N):
        Tmean = evaluate_3rd_derivative(FCnet,Larms[int(arms)][0],Larms[int(arms)][1],Larms[int(arms)][2],N)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*step_size)/( T[arms] + step_size + 0.0 )
        T[arms]          = T[arms]+step_size
        lcb[arms]        = estimate[arms] - np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        ucb[arms]        = estimate[arms] + np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        maxrecord[arms]   = max(maxrecord[arms],estimate[arms])
        minrecord[arms]   = min(minrecord[arms],estimate[arms])
        sigma[arms]       = (maxrecord[arms]-minrecord[arms])/2 
        
    
    chosen_arms=[]
    record_chosen_arms=[]
    k,K=1,K #set pick interactions       
    cnt=0
    
    while k <=K :
        arms_to_pull=choose_arm()
        while arms_to_pull==None:
            
            chosen_arm_pos=np.argmin(ucb)
            chosen_arms.append(chosen_arm_pos) # add the chosen arm to K best arms
            # record the(position, ucb, mean, lcb,T)
            record_chosen_arms.append((chosen_arm_pos,
                                       ucb[chosen_arm_pos],
                                       estimate[chosen_arm_pos],
                                      lcb[chosen_arm_pos],
                                       T[chosen_arm_pos]
                                      ))
            print('chosen arm:',chosen_arm_pos,'strength:',-estimate[chosen_arm_pos], 'iteration:',cnt)
            # set the ucb mean lcb to be a large number, so this arm won't be pulled
            ucb[chosen_arm_pos],estimate[chosen_arm_pos],lcb[chosen_arm_pos]=0.5,0.5,0.5
            arms_to_pull=choose_arm()
            
            k=k+1    
        #print(arms_to_pull)
        pull_arm(arms_to_pull,N)
        cnt=cnt+1
    totaltime=time.time()-st
    print('time:',totaltime)
        
    #reset values    
    for i in record_chosen_arms:
        ucb[i[0]]=i[1]
        estimate[i[0]]=i[2]
        lcb[i[0]]=i[3]
    index_Subset=[]
    for i in chosen_arms:
        index_Subset.append(np.array(Larms[i])) #selected
    
    #save UCB info
    fname=('UCB.txt')
    summary={'ucb':ucb,'estimate':estimate,'lcb':lcb,'T':T}
    f=open(fname,'wb')  
    pickle.dump(summary,f)  
    f.close()    
    
    return index_Subset,totaltime

def detect_4th_UCB(FCnet,X_train,Larms,K):
    np.random.seed(0)
    def evaluate_4th_derivative(net,i,j,k,l,N):
        # eight terms need to be evaluated
        delta=0.5   #np.random.rand()+0.5
        randi=np.random.randint(N)
        xeval = Variable(X_train[randi,:], requires_grad=True)#pick one sample randomly
        xeval=xeval.reshape(1,p) 
        #double side
        f0 = net(xeval-one_hot(i,p)*delta+one_hot(j,p)*delta+one_hot(k,p)*delta+one_hot(l,p)*delta)
        f1 = net(xeval+one_hot(i,p)*delta-one_hot(j,p)*delta+one_hot(k,p)*delta+one_hot(l,p)*delta)
        f2 = net(xeval+one_hot(i,p)*delta+one_hot(j,p)*delta-one_hot(k,p)*delta+one_hot(l,p)*delta)
        f3 = net(xeval-one_hot(i,p)*delta-one_hot(j,p)*delta-one_hot(k,p)*delta+one_hot(l,p)*delta)      
        f4 = net(xeval+one_hot(i,p)*delta+one_hot(j,p)*delta+one_hot(k,p)*delta+one_hot(l,p)*delta)
        f5 = net(xeval-one_hot(i,p)*delta-one_hot(j,p)*delta+one_hot(k,p)*delta+one_hot(l,p)*delta)
        f6 = net(xeval-one_hot(i,p)*delta+one_hot(j,p)*delta-one_hot(k,p)*delta+one_hot(l,p)*delta)
        f7 = net(xeval+one_hot(i,p)*delta-one_hot(j,p)*delta-one_hot(k,p)*delta+one_hot(l,p)*delta)
 
        f8 = net(xeval-one_hot(i,p)*delta+one_hot(j,p)*delta+one_hot(k,p)*delta-one_hot(l,p)*delta)
        f9 = net(xeval+one_hot(i,p)*delta-one_hot(j,p)*delta+one_hot(k,p)*delta-one_hot(l,p)*delta)
        f10 = net(xeval+one_hot(i,p)*delta+one_hot(j,p)*delta-one_hot(k,p)*delta-one_hot(l,p)*delta)
        f11 = net(xeval-one_hot(i,p)*delta-one_hot(j,p)*delta-one_hot(k,p)*delta-one_hot(l,p)*delta)      
        f12 = net(xeval+one_hot(i,p)*delta+one_hot(j,p)*delta+one_hot(k,p)*delta-one_hot(l,p)*delta)
        f13 = net(xeval-one_hot(i,p)*delta-one_hot(j,p)*delta+one_hot(k,p)*delta-one_hot(l,p)*delta)
        f14 = net(xeval-one_hot(i,p)*delta+one_hot(j,p)*delta-one_hot(k,p)*delta-one_hot(l,p)*delta)
        f15 = net(xeval+one_hot(i,p)*delta-one_hot(j,p)*delta-one_hot(k,p)*delta-one_hot(l,p)*delta)
        inter_strength=(f4+f5+f6+f7-f0-f1-f2-f3-f8-f9-f10-f11+f12+f13+f14+f15)/(16*delta**4)
        reward=inter_strength.detach().numpy()**2 #abs()
        return -float(reward)
    print("start dectecting")    
    st=time.time()
    N=X_train.shape[0]
    p=X_train.shape[1]
  
    n=len(Larms)
    Delta = 1.0/n 
    step_size=1
    num_arms=1    #number of arms pulled in one iteration                    
    lcb = np.zeros(n, dtype='float')       #At any point, stores the mu - lower_confidence_interval
    ucb = np.zeros(n, dtype='float')       #At any point, stores the mu + lower_confidence_interval
    T = step_size*np.ones(n, dtype='int')#At any point, stores number of times each arm is pulled

    # initially pull each arm 3 times
    n_init_try=3
    init_data=np.zeros((n_init_try,n))
    for j in range(n_init_try):
        for i in range(n):
            init_data[j,i]=evaluate_4th_derivative(FCnet,Larms[i][0],Larms[i][1],Larms[i][2],Larms[i][3],N)
            
    maxrecord=np.max(init_data,axis=0)
    minrecord=np.min(init_data,axis=0)
    sigma=(maxrecord-minrecord)/2
            
            
    estimate=np.mean(init_data,axis=0)
    lcb = estimate - np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    ucb = estimate + np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    print("Initialization done! initial try"+str(n_init_try)+'times')
    #print(estimate,sigma,np.sqrt(sigma**2*np.log(1/Delta)/step_size))


    def choose_arm():
        n=100 # MAXPULL
        low_lcb_arms = np.argpartition(lcb,num_arms)[:num_arms]
        arms_pulled_morethan_n = low_lcb_arms[ np.where( (T[low_lcb_arms]>=n) & (ucb[low_lcb_arms] != lcb[low_lcb_arms]) ) ]
        if arms_pulled_morethan_n.shape[0]>0:
            # Compute the distance of these arms accurately
            #print('more_than_n')
            #estimate[arms_pulled_morethan_n] = evaluate_2nd_derivative_brute(net,Larms[int(arms_pulled_morethan_n)][0],Larms[int(arms_pulled_morethan_n)][1])
            #T[arms_pulled_morethan_n]  += n
            ucb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            lcb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            return None
        
        if ucb.min() <  lcb[np.argpartition(lcb,1)[1]]: #Exit condition
            return None
        arms_to_pull          = low_lcb_arms[ np.where(T[low_lcb_arms]<n) ]
        return arms_to_pull
    def pull_arm(arms,N):
        Tmean = evaluate_4th_derivative(FCnet,Larms[int(arms)][0],Larms[int(arms)][1],Larms[int(arms)][2],Larms[int(arms)][3],N)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*step_size)/( T[arms] + step_size + 0.0 )
        T[arms]          = T[arms]+step_size
        lcb[arms]        = estimate[arms] - np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        ucb[arms]        = estimate[arms] + np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        maxrecord[arms]   = max(maxrecord[arms],estimate[arms])
        minrecord[arms]   = min(minrecord[arms],estimate[arms])
        sigma[arms]       = (maxrecord[arms]-minrecord[arms])/2 
        
    
    chosen_arms=[]
    record_chosen_arms=[]
    k,K=1,K #set pick interactions       
    cnt=0
    
    while k <=K :
        arms_to_pull=choose_arm()
        while arms_to_pull==None:
            
            chosen_arm_pos=np.argmin(ucb)
            chosen_arms.append(chosen_arm_pos) # add the chosen arm to K best arms
            # record the(position, ucb, mean, lcb,T)
            record_chosen_arms.append((chosen_arm_pos,
                                       ucb[chosen_arm_pos],
                                       estimate[chosen_arm_pos],
                                      lcb[chosen_arm_pos],
                                       T[chosen_arm_pos]
                                      ))
            print('chosen arm:',chosen_arm_pos,'strength:',-estimate[chosen_arm_pos], 'iteration:',cnt)
            # set the ucb mean lcb to be a large number, so this arm won't be pulled
            ucb[chosen_arm_pos],estimate[chosen_arm_pos],lcb[chosen_arm_pos]=0.5,0.5,0.5
            arms_to_pull=choose_arm()
            
            k=k+1    
        #print(arms_to_pull)
        pull_arm(arms_to_pull,N)
        cnt=cnt+1
    totaltime=time.time()-st
    print('time:',totaltime)
        
    #reset values    
    for i in record_chosen_arms:
        ucb[i[0]]=i[1]
        estimate[i[0]]=i[2]
        lcb[i[0]]=i[3]
    index_Subset=[]
    for i in chosen_arms:
        index_Subset.append(np.array(Larms[i])) #selected    
    
    return index_Subset,totaltime

