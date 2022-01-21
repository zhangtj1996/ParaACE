from data import *
from models import *
from utils import *
from detect import detect_Hessian_UCB,detectNID
from sklearn.model_selection import KFold
import os, shutil, pickle
directorys=['St_temp_models','FC_temp_models','Mask_temp_models','Fixup_temp_models','Selected_models','interactions','UCB']  # temp models
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)

    

ctx = mx.gpu(7) if mx.context.num_gpus() > 0 else mx.cpu(0)
df=pd.DataFrame(columns=['Dataset','N_tr', 'seed','FCtest','student_tr','student_te'
                         ])

for datasetindex in range(10):#[0,1,4,5,6,7,8,9]:
    print('dataset')
    dataset=str(datasetindex)+'.csv'
    print('dataset:'+dataset)
    X, y= get_data(dataset)
    

    np.random.seed(0) 
    Xpseudo = np.random.uniform(low=-1, high=1, size=(4000,10))#4000
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
        

        print('start loading FC')
        FCnet=build_FC(train_data,ctx)    # initialize the overparametrized network
        FCnet.load_parameters('Selected_models/FCnet_'+str(datasetindex)+'_seed_'+str(seed))
        loss_function = gluon.loss.L2Loss() 
        testloss=loss_function(FCnet(X_test),y_test)
        FC_test_loss=float(nd.sqrt(2*nd.mean(testloss)).asnumpy()/factor)
        
        print('FCtest:',FC_test_loss)
        
        print('start training student')
        Stnet=build_student(train_data,ctx) 
        Sttr,Stte,Stepoch,StT=train_student(Xpseudo,
            Stnet,FCnet,N,X_test,y_test,X_train,y_train,ctx,factor,epochs=n_epochs,batch_size=batch_size)
        print('Sepoch',Stepoch)
        print('train',Sttr,'test',Stte)
        
        shutil.copy( 'St_temp_models/Stnet_epoch'+str(Stepoch), 'Selected_models/Stnet_'+str(datasetindex)+'_seed_'+str(seed))
        Stnet.load_parameters('Selected_models/Stnet_'+str(datasetindex)+'_seed_'+str(seed))
        
        testloss=loss_function(Stnet(X_test),y_test)
        St_test_loss=float(nd.sqrt(2*nd.mean(testloss)).asnumpy()/factor)
        
        print('Sttest:',St_test_loss)
        
        df=df.append(
                pd.DataFrame(
                {'Dataset':[datasetindex],'N_tr':[N],'seed':[seed],
                 'FCtest':[FC_test_loss],'student_tr':[float(Sttr)],'student_te':[St_test_loss]
                 }
                )
                ,ignore_index=True)
                
        seed += 1
                    
df.to_csv('Teacher_Student+hint.csv')
