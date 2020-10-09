class GRU_Cell:
    def __init__(self,n_a,n_x,acti_func):
        self.__n_a = n_a
        self.__n_x = n_x
        self.__acti = acti_func
        self.__params = {}
        self.__init_param()
        
    def __init_params(self):
        Waa = np.random.randn(self.__n_a,self.__n_a)
        Wax = np.random.randn(self.__n_a,self.__n_x)
        Wya = np.random.randn(self.__n_x,self.__n_a)
        Wua = np.random.randn(self.__n_a,self.__n_a)
        Wux = np.random.randn(self.__n_a,self.__n_x)
        Wra = np.random.randn(self.__n_a,self.__n_a)
        Wrx = np.random.randn(self.__n_a,self.__n_x)
        br = np.zeros((self.__n_a, 1))
        bu = np.zeros((self.__n_a, 1))
        ba = np.zeros((self.__n_a, 1))
        by = np.zeros((self.__n_a, 1))
        self.__params = {'Waa': Waa,
                         'Wax': Wax,
                         'Wya': Wya,
                         'Wua': Wua,
                         'Wux': Wux,
                         'Wra': Wra,
                         'Wrx': Wrx,
                         'br': br,
                         'bu': bu,
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
        elif func == 'sigmoid':
            return np.multiply(X,(1-X))
        
    def cell_forward(self,a_prev,xt):
        zr = np.matmul(self.__params['Wra'],a_prev)+ np.matmul(self.__params['Wrx'],xt) + self.__params['br']
        gr = self.__activation(zr,'sigmoid')
        
        zcc = np.matmul(self.__params['Waa'],np.multiply(gr,a_prev))+ np.matmul(self.__params['Wax'],xt) + self.__params['ba']
        cc = self.__activation(zcc,self.__acti)
        
        zu = np.matmul(self.__params['Wua'],a_prev)+ np.matmul(self.__params['Wux'],xt) + self.__params['bu']
        gu = self.__activation(zu,'sigmoid')
        
        a_next = np.multiply(gu,cc) + np.multiply((1-gu),a_prev)
        
        z_out = np.matmul(self.__params['Wya'],a_next) + self.__params['by']
        a_out = self.__activation(z_out,'softmax')
        
        return a_out,[a_next,a_prev,gu,gr,cc,xt]
    
    def cell_backward(self,y,Y,da_next,cache):
        m = y.shape[1]
        dz_out = y-Y
        da_out = np.matmul(self.__params['Wya'].T,dz_out)
        
        dWya = np.matmul(dz_out,cache[0].T) * (1/m)
        dby = np.sum(dz_out,axis = 1,keepdims = True) * (1/m)
        
        da = da_next + da_out
        dgu = np.multiply(da,(cache[4] - cache[1]))
        dgcc = np.multiply(da,cache[2])
        da_prev = np.multiply((1-cache[2]),da)
        
        dzgu = np.multiply(dgu,self.__d_activation(cache[2],'sigmoid'))
        dWua = np.matmul(dzgu,cache[1].T) * (1/m)
        dWux = np.matmul(dzgu,cache[5].T) * (1/m)
        dbu = np.sum(dzgu,axis = 0,keepdims = True) * (1/m())
        da_prev += np.matmul(self.__params['Wua'].T,dzgu)
        
        dzgcc = np.multiply(dgcc,self.__d_activation(cache[4],self.__acti))
        dWaa = np.matmul(dzgcc,np.multiply(cache[3],cache[4]).T) * (1/m)
        dWax = np.matmul(dzgcc,cache[5].T) * (1/m)
        dba = np.sum(dzgcc,axis = 0,keepdims = True) * (1/m)
        da_prev += np.mutilply(np.matmul(self.__params['Waa'],dzgcc),cache[2])
        
        dgr = np.mutilply(np.matmul(self.__params['Waa'],dzgcc),cache[1])
        dzgr = np.multiply(dgr,self.__d_activation(cache[3],'sigmoid'))
        dWra = np.matmul(dzgr,cache[1].T) * (1/m)
        dWrx = np.matmul(dzgr,cache[5].T) * (1/m)
        dbr = np.sum(dzgr,axis = 0,keepdims = True) * (1/m)
        da_prev += np.matmul(self.__params['Wra'].T,dzgr)
        
        gradients = {'da_prev':da_prev,
                     'dWya':dWya,
                     'dby':dby,
                     'dWua':dWua,
                     'dWux':dWux,
                     'dbu':dbu,
                     'dWaa':dWaa,
                     'dWax':dWax,
                     'dba':dba,
                     'dWra':dWra,
                     'dWrx':dWrx,
                     'dbr':dbr}
        return gradients
    
    def cell_update(self,gradients,alpha):
        self.__params['Waa'] -= alpha * gradients['dWaa']
        self.__params['Wya'] -= alpha * gradients['dWya']
        self.__params['Wax'] -= alpha * gradients['dWax']
        self.__params['Wra'] -= alpha * gradients['dWra']
        self.__params['Wua'] -= alpha * gradients['dWua']
        self.__params['Wrx'] -= alpha * gradients['dWrx']
        self.__params['Wux'] -= alpha * gradients['dWux']
        self.__params['ba'] -= alpha * gradients['ba']
        self.__params['by'] -= alpha * gradients['by']
        self.__params['br'] -= alpha * gradients['br']
        self.__params['bu'] -= alpha * gradients['bu']
