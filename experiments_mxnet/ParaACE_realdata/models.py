import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader, ArrayDataset
import scipy

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
            if i[0]< len(self.masks)-2:
                x=nd.relu(x)
            elif i[0]== len(self.masks)-2:
                self.out=x.T
            else:
                break
        return x.T
    
def build_init_model(index_Subsets,p,train_data,ctx):
    N_blocks=len(index_Subsets)+p # add p with maineffect
    subH1=50
    subH2=8
    n_H1=subH1*N_blocks
    n_H2=subH2*N_blocks
    n_H3=N_blocks
    ### generate masks
    A=np.ones([50,8])
    B=np.ones([8,1])
    A=np.ones([subH1,subH2])
    B=np.ones([subH2,1])
    
    A1=A
    B1=B
    for i in range(N_blocks-1):
        A1=scipy.linalg.block_diag(A1,A)
        B1=scipy.linalg.block_diag(B1,B)
    First=np.zeros([p,n_H1])
    for i in range(len(index_Subsets)):
        First[index_Subsets[i],i*50:(i+1)*50]=1
    ### with maineffect
    for i in range(p):
        First[i,(len(index_Subsets)+i)*50:(len(index_Subsets)+i+1)*50]=1
    masks={}
    masks[0]=nd.array(First).T.as_in_context(ctx)
    masks[1]=nd.array(A1).T.as_in_context(ctx) 
    masks[2]=nd.array(B1).T.as_in_context(ctx) 
    masks[3]=nd.ones([1,N_blocks]).as_in_context(ctx)
    
    net=nn.Sequential()
    net.add(nn.Dense(n_H1,activation='relu'),nn.Dense(n_H2,activation='relu'),nn.Dense(n_H3,activation=None))#,nn.Dense(1, activation=None))
    mx.random.seed(0)
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    #net.initialize(mx.init.Xavier(), ctx=ctx)
    for i, (data, label) in enumerate(train_data):
        aa=net(data.as_in_context(ctx))
        break
    net_initial=net
    Respara=p+len(index_Subsets)
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

    resblock=Residual([15,Respara]) 
    seq=nn.Sequential()
    seq.add(MaskedNet(net_initial,masks),resblock,nn.Dense(1, activation=None))
    mx.random.seed(0)
    seq[1].initialize(mx.init.MSRAPrelu(), ctx=ctx)
    mx.random.seed(0)
    seq[2].initialize(mx.init.MSRAPrelu(), ctx=ctx)
    maskednet = seq
    return maskednet

def build_FC(train_data,ctx):
    n_H1,n_H2,n_H3,n_H4,n_H5=5000,900,400,100,30
    net=nn.Sequential()
    net.add(nn.Dense(n_H1,activation='relu'),nn.Dense(n_H2,activation='relu'),nn.Dense(n_H3,activation='relu'),nn.Dense(n_H4,activation='relu'),nn.Dense(n_H5,activation='relu'),nn.Dense(1, activation=None))
    mx.random.seed(0)
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    #net.initialize(mx.init.Xavier(), ctx=ctx)
    for i, (data, label) in enumerate(train_data):
        aa=net(data.as_in_context(ctx))
        break
    return net
