from data import *
from models import *
from utils import *
from detect import detect_Hessian_UCB,detectNID
from sklearn.model_selection import KFold
import os, shutil, pickle
directorys=['FC_temp_models','Mask_temp_models','Fixup_temp_models','Selected_models','interactions','UCB']  # temp models
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)

    

ctx = mx.gpu(7) if mx.context.num_gpus() > 0 else mx.cpu(0)
df=pd.DataFrame(columns=['Dataset','N_tr', 'seed','DetectT',
                         'Masktr','Maskte','Maskepoch','MaskT','Masksize',
                         'FCtr','FCte','FCepoch','FCT','FCsize',
                       #  'Fixuptr','Fixupte','Fixupepoch','FixupT','Fixupsize',
                         'RFtr','RFte','RFT'
                         ])

for datasetindex in range(10):#[0,1,4,5,6,7,8,9]:
    dataset=str(datasetindex)+'.csv'
    X, y= get_data(dataset)
    

    np.random.seed(0)    
    kf = KFold(n_splits=5,random_state=0,shuffle=True)
    kf.get_n_splits(X)
    seed=0#[0,1,2,3,4]
    chosenarmsList=[]
    for train_index, test_index in kf.split(X):
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
        batch_size=500
        n_epochs=300
        if N<250:
            batch_size=50
        X_train=nd.array(X_tr).as_in_context(ctx)
        y_train=nd.array(y_tr).as_in_context(ctx)
        train_dataset = ArrayDataset(X_train, y_train)
#        num_workers=4
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#,num_workers=num_workers)
        #X_test=nd.array(X_te).as_in_context(ctx)
        #y_test=nd.array(y_te).as_in_context(ctx)
        
        
        RFtr,RFte,RFT=train_RF(X_tr, y_tr.flatten(),X_te,y_te.flatten(), factor, method='reg')
        
       # maskednet=build_init_model(index_Subsets,p,train_data,ctx)
        print('start training FC')
        FCnet=build_FC(train_data,ctx)    # initialize the overparametrized network
        #Masktr,Maskte,Maskepoch=train_masked(maskednet,N,X_test,y_test,train_data,ctx,factor,epochs=100,batch_size=200)
        FCtr,FCte,FCepoch,FCT=train_FC(FCnet,N,X_test,y_test,train_data,ctx,factor,epochs=n_epochs,batch_size=batch_size)
        shutil.copy( 'FC_temp_models/FCnet_epoch'+str(FCepoch), 'Selected_models/FCnet_'+str(datasetindex)+'_seed_'+str(seed))
        FCsize=os.path.getsize('FC_temp_models/FCnet_epoch'+str(FCepoch))/1024
        FCnet.load_parameters('FC_temp_models/FCnet_epoch'+str(FCepoch))
        
        
        index_Subsets,DetectT,chosenarms=detect_Hessian_UCB(FCnet,X_train,ctx,dataset,seed)
        output = open('interactions/dataset'+str(datasetindex)+'_seed_'+str(seed)+'.pkl', 'wb')
        pickle.dump(index_Subsets, output)
        output.close()
        
        chosenarmsList.append(set(chosenarms))
        
        maskednet=build_init_model(index_Subsets,p,train_data,ctx)
        print('start training Mask')
        Masktr,Maskte,Maskepoch,MaskT=train_masked(maskednet,N,X_test,y_test,train_data,ctx,factor,epochs=n_epochs,batch_size=batch_size)
        shutil.copy( 'Mask_temp_models/Maskednet_epoch'+str(Maskepoch), 'Selected_models/Masknet_'+str(datasetindex)+'_seed_'+str(seed))
        Masksize=os.path.getsize('Mask_temp_models/Maskednet_epoch'+str(Maskepoch))/1024
        maskednet.load_parameters('Mask_temp_models/Maskednet_epoch'+str(Maskepoch))
        
       # print('start training Fixup')
       # Fixuptr,Fixupte,Fixupepoch,FixupT,Fixup=train_Fixup(maskednet,N,X_test,y_test,train_data,ctx,factor,epochs=300,batch_size=batch_size)
       # shutil.copy( 'Fixup_temp_models/Fixup_epoch'+str(Fixupepoch), 'Selected_models/Fixup_'+str(datasetindex)+'_seed_'+str(seed))
       # Fixupsize=os.path.getsize('Fixup_temp_models/Fixup_epoch'+str(Fixupepoch))/1024
       # Fixup.load_parameters('Fixup_temp_models/Fixup_epoch'+str(Fixupepoch))
       # print(FCT,MaskT,FixupT)
        print('seed',seed,'Masktr',Masktr,'Maskte',Maskte,'Maskepoch',Maskepoch)
        print('seed',seed,'FCtr',FCtr,'FCte',FCte,'FCepoch',FCepoch)
        df=df.append(
                pd.DataFrame(
                {'Dataset':[datasetindex],'N_tr':[N],'seed':[seed],'DetectT':[DetectT],
                 'Masktr':[Masktr],'Maskte':[Maskte],'Maskepoch':[Maskepoch],'MaskT':[MaskT],'Masksize':[Masksize],
                 'FCtr':[FCtr],'FCte':[FCte],'FCepoch':[FCepoch],'FCT':[FCT],'FCsize':[FCsize],
       #          'Fixuptr':[Fixuptr],'Fixupte':[Fixupte],'Fixupepoch':[Fixupepoch],'FixupT':[FixupT],'Fixupsize':[Fixupsize],
                 'RFtr':[RFtr],'RFte':[RFte],'RFT':[RFT]
                 }
                )
                ,ignore_index=True)
                
        seed += 1
                    
df.to_csv('result_k5.csv')
