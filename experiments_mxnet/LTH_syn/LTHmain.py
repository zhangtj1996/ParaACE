from data import *
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import DataLoader, ArrayDataset
from sklearn.model_selection import KFold
import time
import os, shutil, pickle
directorys=['LTH_temp_models','Selected_models']  # temp models
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)

ctx = mx.gpu(7) if mx.context.num_gpus() > 0 else mx.cpu(0)
df=pd.DataFrame(columns=['Dataset','N_tr', 'p','seed','LTH_tr','LTH_te','LTH_epoch','LTH_T'
                         ])

resultlist=[]    
for datasetindex in range(10):#[0,1,4,5,6,7,8,9]:
    dataset=str(datasetindex)+'.csv'
    X, y= get_data(dataset)
    np.random.seed(0)    
    kf = KFold(n_splits=5,random_state=0,shuffle=True)
    kf.get_n_splits(X)
    seed=0#[0,1,2,3,4]
    for train_index, test_index in kf.split(X):
        print("dataset",datasetindex,"seed:",seed)
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        X_test=nd.array(X_te).as_in_context(ctx)  # Fix test data for all seeds
        y_test=nd.array(y_te).as_in_context(ctx)
        factor=np.max(y_te)-np.min(y_te) #normalize RMSE
        #X_tr, X_te, y_tr, y_te = get_data(0.2,0)
        #selected_interaction = detectNID(X_tr,y_tr,X_te,y_te,test_size,seed)
        #index_Subsets=get_interaction_index(selected_interaction)
        
        N=X_tr.shape[0]
        p=X_tr.shape[1]
        batch_size=400
        #n_epochs=300
        if N<250:
            batch_size=50
        X_train=nd.array(X_tr).as_in_context(ctx)
        y_train=nd.array(y_tr).as_in_context(ctx)
        train_dataset = ArrayDataset(X_train, y_train)
#        num_workers=4
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#,num_workers=num_workers)
        #X_test=nd.array(X_te).as_in_context(ctx)
        #y_test=nd.array(y_te).as_in_context(ctx)
        st=time.time()
        n_H1,n_H2,n_H3,n_H4,n_H5=5000,900,400,100,30
        net=nn.Sequential()
        net.add(nn.Dense(n_H1,activation='relu'),nn.Dense(n_H2,activation='relu'),nn.Dense(n_H3,activation='relu'),nn.Dense(n_H4,activation='relu'),nn.Dense(n_H5,activation='relu'),nn.Dense(1, activation=None))
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        for i, (data, label) in enumerate(train_data):
            aa=net(data.as_in_context(ctx))
            break
        net_initial=net


        def init_masks_percents(net,pList):
            masks={}
            percents={}
            for i in enumerate(net):
                masks[i[0]]=nd.ones(net[i[0]].weight.data().shape).as_in_context(ctx)
                percents[i[0]]=pList[i[0]]
            return masks,percents



        def get_weights(net):
            weights={}
            for i in enumerate(net):
                weights[i[0]]=net[i[0]].weight.data()
            return weights



        def prune_by_percent(percents, masks, final_weights):
            def prune_by_percent_once(percent, mask, final_weight):
                sorted_weights = np.sort(np.abs(final_weight.asnumpy()[mask.asnumpy() == 1]))
                cutoff_index = np.round(percent * sorted_weights.size).astype(int)
                cutoff = sorted_weights[cutoff_index]
                return nd.where(nd.abs(final_weight) <= cutoff, nd.zeros(mask.shape).as_in_context(ctx), mask)

            new_masks = {}
            for k, percent in percents.items():
                new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k])
            return new_masks



        class MaskedNet(nn.Block):
            # For FC network
            def __init__(self,net,masks,**kwargs):
                super(MaskedNet, self).__init__(**kwargs)  
                self.net=net
                self.masks=masks

            def forward(self, x):
                x=x.T
                for i in enumerate(self.net):
                    x=nd.dot(self.net[i[0]].weight.data()*self.masks[i[0]],x)
                    x=x+self.net[i[0]].bias.data().reshape((self.net[i[0]].bias.data().size,1))
                    if i[0]< len(self.masks)-1:
                        x=nd.relu(x)
                    else:
                        break
                return x.T

        ### initialize Masknet    
        masks,percents=init_masks_percents(net,[0.2,0.2,0.2,0.2,0.2,0])
        maskednet=MaskedNet(net_initial,masks)


        trainer = gluon.Trainer(
            params=maskednet.collect_params(),
            optimizer = 'adam',

        )

        metric = mx.metric.Accuracy()
        loss_function = gluon.loss.L2Loss() 


        epochs = 502
        num_batches = N / batch_size
        trainerr=[]
        testerr=[]
        ep=[]  
        test_temp=[]
        for e in range(epochs):
            cumulative_loss = 0
            for i, (data, label) in enumerate(train_data):
                #print(i)
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                with autograd.record():
                    output = maskednet(data)
                    loss=loss_function(output, label)
                loss.backward()

                trainer.step(batch_size)#,ignore_stale_grad=True) ###ignore
                cumulative_loss += nd.mean(loss).asscalar()

                
                
            if e%20==0 and e!=0:
                final_w=get_weights(maskednet.net)        
                masks=prune_by_percent(percents, masks, final_w)  # Update Masks
                maskednet=MaskedNet(net_initial,masks)            # Reset Network with new mask
                resultlist.append(min(test_temp))
                print("===== prune =====",min(test_temp))
                test_temp=[]
                
            #print("Epoch %s, loss: " % (e))
            f_loss=np.sqrt(2*cumulative_loss / num_batches)/factor

            testloss=loss_function(maskednet(nd.array(X_test).as_in_context(ctx)),y_test)
            test_loss=nd.sqrt(2*nd.mean(testloss)).asnumpy()/factor
            test_temp.append(test_loss)

            #print('train_loss:',f_loss,'test_loss:',test_loss)
            trainerr.append(f_loss)
            testerr.append(test_loss)
            maskednet.save_parameters('LTH_temp_models/LTHnet_epoch'+str(e))
            ep.append(e)
        totaltime=time.time()-st     

        LTHtr,LTHte,LTHepoch,LTHT=trainerr[np.argmin(testerr)],float( min(testerr)),np.argmin(testerr), totaltime
    
    


        df=df.append(
                pd.DataFrame(
                {'Dataset':[datasetindex],'N_tr':[N],'p':[p],'seed':[seed],
                 'LTH_tr':[LTHtr],'LTH_te':[LTHte],'LTH_epoch':[LTHepoch],'LTH_T':[LTHT]
                 }
                )
                ,ignore_index=True)
                
        seed += 1
resultnpy=np.array(resultlist)
np.save("result.npy",resultnpy)
df.to_csv('result.csv')
