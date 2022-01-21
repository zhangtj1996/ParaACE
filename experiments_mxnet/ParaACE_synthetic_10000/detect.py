import numpy as np
import math
import bisect
import operator
import tensorflow as tf
from sklearn.preprocessing import Normalizer,StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import itertools
import matplotlib.pyplot as plt
from mxnet import nd
import pickle




def evaluate_2nd_derivative_brute(net,x_tr,i,k,p):
    delta=1
    xeval = x_tr#Variable(x_tr, requires_grad=True) #pick one sample
    f0 = net(xeval)
    fi =net(xeval+one_hot(i,p)*delta)
    fik =net(xeval+one_hot(i,p)*delta+one_hot(k,p)*delta)
    fk=net(xeval+one_hot(k,p)*delta)
    inter_strength=(fik-fi-fk+f0)/delta**2
    #print(inter_strength)
    reward=inter_strength**2 #abs()
    meanreward=(np.mean(reward.asnumpy()))
    return -float(meanreward)





def detect_Hessian_bruteforce(FCnet,X_train):
    N=X_train.shape[0]
    p=X_train.shape[1]
    Larms=[]
    for i in itertools.combinations(range(p),2):
        Larms.append(i)
    
    st=time.time()
    matrix=np.zeros([p,p])
    for i in Larms:
        matrix[i[0],i[1]]=evaluate_2nd_derivative_brute(FCnet,X_train,i[0],i[1],p)
        matrix[i[1],i[0]]=matrix[i[0],i[1]]
    plt.matshow(matrix)
    print('time:', time.time()-st)
    return matrix
    
def detect_Hessian_UCB(FCnet,X_train,ctx,dataset,seed):
    np.random.seed(0)
    def one_hot(i,p):
        return nd.one_hot(nd.array([i]),p).as_in_context(ctx)
    def evaluate_2nd_derivative(net,i,k,N):
        delta=1.5#np.random.rand()+0.5
        j=np.random.randint(N)
        xeval = X_train[j,:].as_in_context(ctx).reshape(1,p) #pick one sample
        ###one side
        #f0 = net(xeval)
        #fi =net(xeval+one_hot(i,p)*delta)
        #fik =net(xeval+one_hot(i,p)*delta+one_hot(k,p)*delta)
        #fk=net(xeval+one_hot(k,p)*delta)
        
        ### two side
        f0 = net(xeval-one_hot(i,p)*delta-one_hot(k,p)*delta)
        fi =net(xeval+one_hot(i,p)*delta-one_hot(k,p)*delta)
        fik =net(xeval+one_hot(i,p)*delta+one_hot(k,p)*delta)
        fk=net(xeval+one_hot(k,p)*delta-one_hot(i,p)*delta)
        
        inter_strength=(fik-fi-fk+f0)/(4*delta**2)
        reward=inter_strength.asnumpy()**2 #abs()
        return -float(reward)
    print("start dectecting")    
    st=time.time()
    N=X_train.shape[0]
    p=X_train.shape[1]
    Larms=[]
    for i in itertools.combinations(range(p),2):
        Larms.append(i)
    
    if p*(p-1)/2<20:
        indexSubsets=[]
        for i in range(len(Larms)):
            indexSubsets.append(np.array(Larms[i]))
        totaltime=time.time()-st
        chosen_arms=range(int(p*(p-1)/2))
        return indexSubsets,totaltime,chosen_arms
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
    k,K=1,20 #set pick interactions       
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
            with open('log/'+dataset+"_"+str(seed)+".txt","a") as f:
                f.writelines('chosen arm:'+str(chosen_arm_pos)+'strength:'+str(-estimate[chosen_arm_pos])+ 'iteration:'+str(cnt)+"\n")
            
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
        
    fname=('UCB/'+dataset+str(seed)+'UCB.txt')
    summary={'ucb':ucb,'estimate':estimate,'lcb':lcb,'T':T}
    f=open(fname,'wb')  
    pickle.dump(summary,f)  
    f.close()    
    return index_Subset,totaltime,chosen_arms





def detectNID(tr_x,tr_y,te_x,te_y,test_size,seed):
    use_main_effect_nets = True # toggle this to use "main effect" nets
    # Parameters
    learning_rate = 0.01
    num_epochs = 200
    batch_size = int(tr_x.shape[0]/10)#100
    display_step = 100
    l1_const = 5e-5 #-5
    #num_samples = 30000 #30k datapoints, split 1/3-1/3-1/3
    
    # Network Parameters
    n_hidden_1 = 140 # 1st layer number of neurons
    n_hidden_2 = 100 # 2nd layer number of neurons
    n_hidden_3 = 60 # 3rd "
    n_hidden_4 = 20 # 4th "
    n_hidden_uni = 10
    num_input = tr_x.shape[1] # simple synthetic example input dimension
    num_output = 1 # regression or classification output dimension
    
    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])
    
    tf.set_random_seed(0)
    np.random.seed(0)
    
    #tr_x, va_x, te_x, tr_y, va_y, te_y = get_data()
    tr_size = tr_x.shape[0]
    
    # access weights & biases
    weights = {
        'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_4, num_output], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3], 0, 0.1)),
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.truncated_normal([num_output], 0, 0.1))
    }
    
    def get_weights_uninet():
        weights = {
            'h1': tf.Variable(tf.truncated_normal([1, n_hidden_uni], 0, 0.1)),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_uni, n_hidden_uni], 0, 0.1)),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_uni, n_hidden_uni], 0, 0.1)),
            'out': tf.Variable(tf.truncated_normal([n_hidden_uni, num_output], 0, 0.1))
        }
        return weights
    
    def get_biases_uninet():
        biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_uni], 0, 0.1)),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_uni], 0, 0.1)),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_uni], 0, 0.1))
        }
        return biases
    
    # Create model
    def normal_neural_net(x, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))    
        out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
        return out_layer
    
    def main_effect_net(x, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))    
        out_layer = tf.matmul(layer_3, weights['out'])
        return out_layer
    
    # L1 regularizer
    def l1_norm(a): return tf.reduce_sum(tf.abs(a))
    # Construct model
    net = normal_neural_net(X, weights, biases)
    
    if use_main_effect_nets:  
        me_nets = []
        for x_i in range(num_input):
            me_net = main_effect_net(tf.expand_dims(X[:,x_i],1), get_weights_uninet(), get_biases_uninet())
            me_nets.append(me_net)
        net = net + sum(me_nets)
    
    # Define optimizer
    loss_op = tf.losses.mean_squared_error(labels=Y, predictions=net)
    # loss_op = tf.sigmoid_cross_entropy_with_logits(labels=Y,logits=net) # use this in the case of binary classification
    sum_l1 = tf.reduce_sum([l1_norm(weights[k]) for k in weights])
    loss_w_reg_op = loss_op + l1_const*sum_l1 
    
    batch = tf.Variable(0)
    decaying_learning_rate = tf.train.exponential_decay(learning_rate, batch*batch_size, tr_size, 0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate).minimize(loss_w_reg_op, global_step=batch)
    init = tf.global_variables_initializer()
    n_batches = tr_size//batch_size
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.25
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    print('Initialized')
    
    for epoch in range(num_epochs):
    
        batch_order = list(range(n_batches))
        np.random.shuffle(batch_order)
    
        for i in batch_order:
            batch_x = tr_x[i*batch_size:(i+1)*batch_size]
            batch_y = tr_y[i*batch_size:(i+1)*batch_size]
            _, lr = sess.run([optimizer,decaying_learning_rate], feed_dict={X:batch_x, Y:batch_y})
    
        if (epoch+1) % 50 == 0:
            tr_mse = sess.run(loss_op, feed_dict={X:tr_x, Y:tr_y})
           # va_mse = sess.run(loss_op, feed_dict={X:va_x, Y:va_y})
            te_mse = sess.run(loss_op, feed_dict={X:te_x, Y:te_y})
            print('Epoch', epoch+1)
            print('\t','train rmse', math.sqrt(tr_mse), 'test rmse', math.sqrt(te_mse))
            print('\t','learning rate', lr)
            
    print('done')
    
    def preprocess_weights(w_dict):
        hidden_layers = [int(layer[1:]) for layer in w_dict.keys() if layer.startswith('h')]
        output_h = ['h' + str(x) for x in range(max(hidden_layers),1,-1)]
        w_agg = np.abs(w_dict['out'])
        w_h1 = np.abs(w_dict['h1'])
    
        for h in output_h:
            w_agg = np.matmul( np.abs(w_dict[h]), w_agg)
    
        return w_h1, w_agg 
    
    def get_interaction_ranking(w_dict):
        xdim = w_dict['h1'].shape[0]
        w_h1, w_agg = preprocess_weights(w_dict)
            
        # rank interactions
        interaction_strengths = dict()
    
        for i in range(len(w_agg)):
            sorted_fweights = sorted(enumerate(w_h1[:,i]), key=lambda x:x[1], reverse = True)
            interaction_candidate = []
            weight_list = []       
            for j in range(len(w_h1)):
                bisect.insort(interaction_candidate, sorted_fweights[j][0]+1)
                weight_list.append(sorted_fweights[j][1])
                if len(interaction_candidate) == 1:
                    continue
                interaction_tup = tuple(interaction_candidate)
                if interaction_tup not in interaction_strengths:
                    interaction_strengths[interaction_tup] = 0
                inter_agg = min(weight_list)      
                interaction_strengths[interaction_tup] += np.abs(inter_agg*np.sum(w_agg[i]))
            
        interaction_sorted = sorted(interaction_strengths.items(), key=operator.itemgetter(1), reverse=True)
    
        # forward prune the ranking of redundant interactions
        interaction_ranking_pruned = []
        existing_largest = []
        for i, inter in enumerate(interaction_sorted):
            if len(interaction_ranking_pruned) > 20000: break
            skip = False
            indices_to_remove = set()
            for inter2_i, inter2 in enumerate(existing_largest):
                # if this is not the existing largest
                if set(inter[0]) < set(inter2[0]):
                    skip = True
                    break
                # if this is larger, then need to recall this index later to remove it from existing_largest
                if set(inter[0]) > set(inter2[0]):
                    indices_to_remove.add(inter2_i)
            if skip:
                assert len(indices_to_remove) == 0
                continue
            prevlen = len(existing_largest)
            existing_largest[:] = [el for el_i, el in enumerate(existing_largest) if el_i not in indices_to_remove]
            existing_largest.append(inter)
            interaction_ranking_pruned.append((inter[0], inter[1]))
    
            curlen = len(existing_largest)
    
        return interaction_ranking_pruned
    
    def get_pairwise_ranking(w_dict):
        xdim = w_dict['h1'].shape[0]
        w_h1, w_agg = preprocess_weights(w_dict)
    
        input_range = range(1,xdim+1)
        pairs = [(xa,yb) for xa in input_range for yb in input_range if xa != yb]
        for entry in pairs:
            if (entry[1], entry[0]) in pairs:
                pairs.remove((entry[1],entry[0]))
    
        pairwise_strengths = []
        for pair in pairs:
            a = pair[0]
            b = pair[1]
            wa = w_h1[a-1].reshape(w_h1[a-1].shape[0],1)
            wb = w_h1[b-1].reshape(w_h1[b-1].shape[0],1)
            wz = np.abs(np.minimum(wa , wb))*w_agg
            cab = np.sum(np.abs(wz))
            pairwise_strengths.append((pair, cab))
    #     list(zip(pairs, pairwise_strengths))
    
        pairwise_ranking = sorted(pairwise_strengths,key=operator.itemgetter(1), reverse=True)
    
        return pairwise_ranking
    w_dict = sess.run(weights)
    # Get the effect interaction
    def get_npy_from_NN(threhold):
        
        all_interaction = get_interaction_ranking(w_dict)
    
        selected_interaction = []
        for _i in range(len(all_interaction)):
            if _i < threhold:
                selected_interaction.append(all_interaction[_i][0])
            else:
                break
        print(len(selected_interaction))
        print(selected_interaction)
        
        return selected_interaction
    
    threhold = 30
    selected_interaction = get_npy_from_NN(threhold)
    numpy_data = np.array(selected_interaction)
    np.save('interactions/NN_data_for_ACE_hw5_'+str(test_size)+'_' +str(seed)+'.npy', numpy_data)
    return numpy_data
