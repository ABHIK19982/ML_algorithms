import numpy as np

class RNN_Cell:
    def __init__(self,n_a,n_x,acti_func):
        self.__n_a = n_a
        self.__n_x = n_x
        self.__acti = acti_func
        self.__params = {}
        self.__init_param()
    
    def __init_param(self):
        np.random.seed(1)
        
        Waa = np.random.randn(self.__n_a,self.__n_a)
        Wax = np.random.randn(self.__n_a,self.__n_x)
        Wya = np.random.randn(self.__n_x,self.__n_a)
        ba = np.zeros((self.__n_a,1))
        by = np.zeros((self.__n_x,1))
        self.__params = {'Waa': Waa,
                         'Wax': Wax,
                         'Wya': Wya,
                         'ba': ba,
                         'by': by}
    
    def __activation(self,X,func):
        if(func == 'identity'):
            return X
        elif(func == 'tanh'):
            return np.tanh(X)
        elif(func == 'relu'):
            return np.maximum(X,0)
        elif(func == 'softmax'):
            val = np.sum(np.exp(X),axis = 0,keepdims = True)

            val = np.exp(X)/val
            return val
    
    def __d_activation(self,X,func):
        if(func == 'identity'):
            pass
        elif(func == 'tanh'):
            return (1-X**2)
        elif func == 'relu':
            return (X>0)
    
    def cell_forward(self,a_prev,xt):
        z_next = np.matmul(self.__params['Waa'],a_prev) + np.matmul(self.__params['Wax'],xt) + self.__params['ba']
        a_next = self.__activation(z_next,self.__acti)
        
        z_out = np.matmul(self.__params['Wya'],a_next) + self.__params['by']
        y_out = self.__activation(z_out,'softmax')
        
        return y_out,[a_next,a_prev,xt]
    
    def cell_backward(self,da_next,y,Y,cache):
        m = y.shape[1]
        
        dz_out = y-Y
        da_out = np.matmul(self.__params['Wya'].T,dz_out)
        
        dWya = np.matmul(dz_out,cache[0].T) * (1/m)
        dby = np.sum(dz_out,axis = 1,keepdims = True) * (1/m)
        
        da = da_next + da_out
        dz = np.multiply(da,self.__d_activation(cache[0],self.__acti))
        da_prev = np.matmul(self.__params['Waa'].T,dz)
        
        dWaa = np.matmul(dz,cache[1].T) * (1/m)
        dWax = np.matmul(dz,cache[2].T) * (1/m)
        dba = np.sum(dz,axis = 1,keepdims = True) * 1/m
        
        gradients = {'da_prev':da_prev,
                     'dWya':dWya,
                     'dWaa':dWaa,
                     'dWax':dWax,
                     'dba':dba,
                     'dby':dby}
        return gradients
    
    def cell_update(self,gradients,alpha):
        self.__params['Waa'] -= alpha * gradients['dWaa']
        self.__params['Wya'] -= alpha * gradients['dWya']
        self.__params['Wax'] -= alpha * gradients['dWax']
        self.__params['ba'] -= alpha * gradients['ba']
        self.__params['by'] -= alpha * gradients['by']
    
    

