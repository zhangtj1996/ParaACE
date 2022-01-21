from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_RF(X_tr, y_tr,X_te,y_te, factor, method='reg'):
    st=time.time()    
    if method == 'reg':
    
        RF_reg = RandomForestRegressor()
        RF_reg.fit(X_tr, y_tr)
        
        y_tr_pred = RF_reg.predict(X_tr)
        y_te_pred = RF_reg.predict(X_te)
        
        tr_nrmse = np.sqrt(np.mean(np.square(y_tr - y_tr_pred)))/factor
        te_nrmse = np.sqrt(np.mean(np.square(y_te - y_te_pred)))/factor
    RFT=time.time()-st
        

#            
#        if method == 'cla':
#            
#            RF_cla = RandomForestClassifier()
#            RF_cla.fit(X_tr, y_tr)
#            
#            y_tr_pred = RF_cla.predict(X_tr)
#            y_te_pred = RF_cla.predict(X_te)
#        
#            tr_acc = accuracy_score(y_tr,y_tr_pred)
#            te_acc = accuracy_score(y_te,y_te_pred)
#    
#            res[i,:] = [tr_acc, te_acc]
        
    return tr_nrmse,te_nrmse,RFT



def train_masked(maskednet,N,X_test,y_test,train_data,ctx,factor,epochs=100,batch_size=200):
    st=time.time()
    trainer = gluon.Trainer(
        params=maskednet.collect_params(),
        optimizer = 'adam',
    )
    
#    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.L2Loss() 
    epochs = epochs
    num_batches = N / batch_size
    trainerr=[]
    testerr=[]
    ep=[]        
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            #print(i)
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = maskednet(data)
                loss=loss_function(output, label)
                loss1=loss#+0.005*nd.sum(nd.abs(maskednet.net[3].weight.data()))
            loss1.backward()
          
            trainer.step(batch_size)#,ignore_stale_grad=True) ###ignore
            cumulative_loss += nd.mean(loss).asscalar()
        
        if e%1==0:
            #print("Epoch %s: " % (e))
            
            f_loss=np.sqrt(2*cumulative_loss / num_batches)/factor#nd.mean(loss1).asscalar()
            testloss=loss_function(maskednet(nd.array(X_test).as_in_context(ctx)),y_test)
            test_loss=nd.sqrt(2*nd.mean(testloss)).asnumpy()/factor
           # print('train_loss:',f_loss,'test_loss:',test_loss)
            trainerr.append(f_loss)
            testerr.append(test_loss)
            maskednet.save_parameters('Mask_temp_models/Maskednet_epoch'+str(e))
            ep.append(e)
    totaltime=time.time()-st        
    return trainerr[np.argmin(testerr)],float( min(testerr)),np.argmin(testerr), totaltime

def train_FC(FCnet,N,X_test,y_test,train_data,ctx,factor,epochs=100,batch_size=200):
    st=time.time()
    trainer = gluon.Trainer(
        params=FCnet.collect_params(),
        optimizer = 'adam',
    )
    
#    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.L2Loss() 
    epochs = epochs
    num_batches = N / batch_size
    trainerr=[]
    testerr=[]
    ep=[]        
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            #print(i)
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = FCnet(data)
                loss=loss_function(output, label)
            loss.backward()
          
            trainer.step(batch_size)#,ignore_stale_grad=True) ###ignore
            cumulative_loss += nd.mean(loss).asscalar()
        
        if e%1==0:
            #print("Epoch %s: " % (e))
            
            f_loss=np.sqrt(2*cumulative_loss / num_batches)/factor#nd.mean(loss1).asscalar() *2 is from the lossfunction
            testloss=loss_function(FCnet(nd.array(X_test).as_in_context(ctx)),y_test)
            test_loss=nd.sqrt(2*nd.mean(testloss)).asnumpy()/factor
            #print('train_loss:',f_loss,'test_loss:',test_loss)
            trainerr.append(f_loss)
            testerr.append(test_loss)
            FCnet.save_parameters('FC_temp_models/FCnet_epoch'+str(e))
            ep.append(e)
    
    totaltime=time.time()-st        
    return trainerr[np.argmin(testerr)],float( min(testerr)),np.argmin(testerr),totaltime

def train_Fixup(maskednet,N,X_test,y_test,train_data,ctx,factor,epochs=100,batch_size=200):
    st=time.time()
    Respara=maskednet[1].weight.data().shape[1]
    class Residual(nn.Block):
        def __init__(self, layerlist, **kwargs):
            super(Residual, self).__init__(**kwargs)
            #print(layerlist)
            for i in range(len(layerlist)):
                #print(i,layerlist[i])
                exec('self.dense'+str(i)+"=nn.Dense(layerlist[i],activation='relu')")
          
        def forward(self, X):
            Y = (self.dense1(self.dense0(X)))
            #Y = self.bn2(self.conv2(Y))
            #if self.conv3:
             #   X = self.conv3(X)
            return (Y + X)

    resblock=Residual([50,Respara]) ## set fixup
    Fixup=nn.Sequential()
    Fixup.add(maskednet[0])
    Fixup.add(resblock,nn.Dense(1,activation=None))
    mx.random.seed(0)
    for i in range(2):
        Fixup[i+1].initialize(mx.init.Xavier(), ctx=ctx,force_reinit=True)
    for param in Fixup[0].collect_params().values():
        param.grad_req = 'null'
     
    trainer2 = gluon.Trainer(
        params=Fixup.collect_params(),
        optimizer = 'adam',
    )
    
    loss_function = gluon.loss.L2Loss() 
    
    num_batches = N / batch_size
    trainerr=[]
    testerr=[]
    ep=[]        
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            #print(i)
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = Fixup(data)
                loss=loss_function(output, label)
            loss.backward()
          
            trainer2.step(batch_size)#,ignore_stale_grad=True) ###ignore
            cumulative_loss += nd.mean(loss).asscalar()
        
        if e%1==0:
    #        final_w=get_weights(maskednet.net)        
    #        masks=prune_by_percent(percents, masks, final_w)  # Update Masks
    #        maskednet=MaskedNet(net_initial,masks)            # Reset Network with new mask
    #        print("Epoch %s: " % (e))
    #        f_loss=nd.mean(loss).asscalar()
    #        testloss=loss_function(maskednet(nd.array(X_test).as_in_context(ctx)),y_test)
    #        print('train_loss:',f_loss,'test_loss:',nd.mean(testloss).asnumpy())
            #print(e)
            f_loss=np.sqrt(2*cumulative_loss / num_batches)/factor#nd.mean(loss1).asscalar()
            testloss=loss_function(Fixup(nd.array(X_test).as_in_context(ctx)),y_test)
            test_loss=nd.sqrt(nd.mean(testloss)*2).asnumpy()/factor
            #print('train_loss:',f_loss,'test_loss:',test_loss)
            trainerr.append(f_loss)
            Fixup.save_parameters('Fixup_temp_models/Fixup_epoch'+str(e))
            testerr.append(test_loss)
            ep.append(e)
    totaltime=time.time()-st
    return trainerr[np.argmin(testerr)],float(min(testerr)),np.argmin(testerr),totaltime,Fixup

