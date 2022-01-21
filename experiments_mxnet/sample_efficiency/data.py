import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_test_data(dataset,test_size = 0.2, random_state = 0):    
    data = pd.read_csv("regression/"+dataset, header=0)
    y, X = np.array(data.iloc[:,-1]).reshape(-1,1), np.array(data.iloc[:,0:-1])
    print(X.shape); print(y.shape)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, 
     #                                                   test_size = 0.2, random_state=0)
    tr_x, te_x, tr_y, te_y = train_test_split(X, y, test_size = test_size, random_state = random_state)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(tr_x)
    scaler_y.fit(tr_y)

    tr_x, te_x = scaler_x.transform(tr_x), scaler_x.transform(te_x)
    tr_y, te_y = scaler_y.transform(tr_y), scaler_y.transform(te_y)
    return tr_x, te_x, tr_y,  te_y

def get_data(dataset):    
    data = pd.read_csv("regression/"+dataset, header=0)
    y, X = np.array(data.iloc[:,-1]).reshape(-1,1), np.array(data.iloc[:,0:-1])
    print(X.shape); print(y.shape)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X)
    scaler_y.fit(y)

    X = scaler_x.transform(X)
    y = scaler_y.transform(y)
    return X,y



def get_interaction_index(index_Subsets):
    
    ## Load the index subsets from NN
    #index_Subsets = np.load("firsthuawei/NN_data_for_ACE_hw"+str(dataset)+".npy",allow_pickle=True)
    index_Subsets = index_Subsets.tolist()
    index_Subsets = [np.array(Si)-1 for Si in index_Subsets]
    print(index_Subsets)
    print(len(index_Subsets))
    return index_Subsets
