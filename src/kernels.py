import torch
import numpy as np
import scipy.linalg
import scipy.spatial.distance as sd
import math
import sklearn.metrics.pairwise as pw

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#print(device)



class Kernel(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, X=None, Y=None, pair_dist=None, grad=True, numpy=False):
        '''
        To compute the kernel value for each entrance of pair_dist if provided.
        Otherwise computes firt the pairwise distance matrix between X and X,
        or X and Y depending on if both are provided or not.

        Parameters
        -----------
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. Needs to be provided
            if pair_dist is None. If Y is None, then the Gram matrix of X and X is 
            computed (default is None).
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            and pairwise_dist is None, then Gram matrix between X and Y is
            computed (default is None).
        pair_dist : (n,n) numpy ndarray or torch.Tensor
            Distance between pairs x_i and y_j, if provided (default is None).
        grade : Boolean
            True if gradient should be computed (default is True).
        numpy : Boolean
            True if should return numpy instead of torch.Tensor (defualt is False). 

        Raises
        -------
        NotImplementedError 
            Needs to be implemented in subclasses.
        '''
        raise NotImplementedError("Subclasses should implement this!")

    def forward_w_o_pair_dist(self, X, Y=None, grad=False, numpy=True):
        '''
        If one wish to compue Gram marix without computing the pairwise distance first.
        Either between X and itself, or X and Y if provided.

        Parameters
        ----------
        X : (n,n) numpy ndarray
            Matrix with points to compute Gram matrix with.
            If Y is None, then the Gram matrix of X and X is computed.
        Y : (n,n) numpy ndarray
            Matrix with points to compute Gram matarix of X and Y with,
            if provided (default is None)
        grade : Boolean
            True if gradient should be computed (default is False).
        numpy : Boolean
            True if should return numpy instead of torch.Tensor (defualt is True).

        Returns
        -------
        Gram matrix : (n,n) ndarray or torch.Tensor
            Gram matrix computed between X and X, or X and Y if provided.
        '''
        return self.forward(self.pairwise_dist(X,Y), grad, numpy)

    def pairwise_dist(self, X,Y=None):
        '''
        Computes pairwise distance between X and X, or X and Y if provided.
        X : (n,n) numpy ndarray
            Matrix with points to compute pairwise distance with.
            If Y is None, then the pairwise distance matrix of X and X is computed.
        Y : (n,n) numpy ndarray
            Matrix with points to compute pairwise distance matarix of X and Y with,
            if provided (default is None)
        
        Return
        ------
        pairwise distance : (n,n) ndarray
            The pairwise distance between X and X, or X and Y if provided.
        '''
       #print("pairwise func x", X.shape)
       # print("pairwise func y", Y.shape)
        if Y is None:
            return np.square(sd.squareform(sd.pdist(X.T)))
   
        return np.square(sd.cdist(X.T,Y.T))



class RBFKernel(Kernel):

    def __init__(self, params=None):
        super().__init__()
        self.params = torch.nn.ParameterList(params)
    
    def forward(self, X, Y=None,pair_dist=None, grad=True, numpy=False):
        '''
        To compute the value using the RBF kernel, for each entrance of pair_dist if provided.
        Otherwise computes firt the pairwise distance matrix between X and X,
        or X and Y depending on if both are provided or not.

        Parameters
        -----------
        pair_dist : (n,n) numpy ndarray or torch.Tensor
            Distance between pairs x_i and y_j, if provided (default is None).
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. Needs to be provided
            if pair_dist is None. If Y is None, then the Gram matrix of X and X is 
            computed (default is None).
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            and pairwise_dist is None, then Gram matrix between X and Y is
            computed (default is None).
        grade : Boolean
            True if gradient should be computed (default is True).
        numpy : Boolean
            True if should return numpy instead of torch.Tensor (defualt is False).

        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix using the pair_dist, or between X and X, or X and Y depending
            on what is None.

        '''

        #assert X is not None, 'X must be provided when pair_dist is None.'
            #made this change here -CNB added not
        if pair_dist is None:
            if Y is not None:
                pair_dist = self.pairwise_dist(X,Y)
            else:
                pair_dist = self.pairwise_dist(X)

        if not grad: 
            with torch.no_grad():
                pair_dist=torch.tensor(pair_dist)
                ret = torch.exp(-(1/(2*torch.square(self.params[0]))) * pair_dist)
            return ret.cpu().detach().numpy() if numpy else ret
        
        pair_dist=torch.tensor(pair_dist)
        ret = torch.exp(-(1/(2*torch.square(self.params[0]))) * pair_dist)
        return ret.cpu().detach().numpy() if numpy else ret

                         
class PolynomialKernel(Kernel):
    def __init__(self):
        super().__init__()

    
    def forward(self, X, Y=None,grad=False, numpy=False):
        '''
        To compute the value using the Linear none paramaterized kernel.

        Parameters
        -----------
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix.
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            then Gram matrix between X and Y is computed (default is None).
        
        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix between X and X, or X and Y depending
            on whether or not Y is provided.
        '''
        
    
        
        if Y is None:  
            out= (X.T@X +1)**3
           
            return out
             
        out=(X.T@Y +1)**3
       
        return out 
        
class SigmoidKernel(Kernel):
    def __init__(self, params):
        super().__init__()  
        self.params = torch.nn.ParameterList(params)
      
       

    def forward(self, X, Y=None, grad=True, numpy=False):
        '''
        To compute the value using the Linear none paramaterized kernel.

        Parameters
        -----------
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix.
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            then Gram matrix between X and Y is computed (default is None).
        
        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix between X and X, or X and Y depending
            on whether or not Y is provided.
        '''
        
        
        if Y is None:
            x=torch.tensor(X)
            out=torch.tanh(torch.mul(self.params[0],torch.tensor(X.T@X))+self.params[1])
            return out.cpu().detach().numpy() if numpy else out #sigmoid_kernel(X,X).cpu().detach().numpy() if numpy else sigmoid_kernel(X,X).cpu()
        else:
            x=torch.tensor(X)
            y=torch.tensor(Y)
            out=torch.tanh(torch.mul(self.params[0],torch.tensor(X.T@Y))+self.params[1])
            return out.cpu().detach().numpy() if numpy else out# sigmoid_kernel(X,Y).cpu().detach().numpy() if numpy else sigmoid_kernel(X,Y).cpu()
    


class ComplexKernel(Kernel):
    def __init__(self, params):
        super().__init__()
        self.params = torch.nn.ParameterList(params)
    
    
    def forward(self, X, Y=None, pair_dist=None, grad=True, numpy=False):
        '''
        To compute the value using the RBF kernel, for each entrance of pair_dist if provided.
        Otherwise computes firt the pairwise distance matrix between X and X,
        or X and Y depending on if both are provided or not.

        Parameters
        -----------
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. Needs to be provided
            if pair_dist is None. If Y is None, then the Gram matrix of X and X is 
            computed (default is None).
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            and pairwise_dist is None, then Gram matrix between X and Y is
            computed (default is None).
        pair_dist : (n,n) numpy ndarray or torch.Tensor
            Distance between pairs x_i and y_j, if provided (default is None).
        grade : Boolean
            True if gradient should be computed (default is True).
        numpy : Boolean
            True if should return numpy instead of torch.Tensor (defualt is False).

        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix using the pair_dist, or between X and X, or X and Y depending
            on what is None.

        '''

        #assert X is not None, 'X must be provided when pair_dist is None.'
            #made this change here -CNB added not

        
        if Y is not None:
            pair_dist = self.pairwise_dist(X,Y)
        else:
            pair_dist = self.pairwise_dist(X)

        if Y is None: 
            if not grad: 
                with torch.no_grad():
                    ret = torch.exp(-(1/(2*torch.square(self.params[0]))) * torch.tensor(pair_dist))+torch.tensor((X.T@X +1)**3)
                return ret.cpu().detach().numpy() if numpy else ret
            pair_dist=torch.tensor(pair_dist)    
            ret = torch.exp(-(1/(2*torch.square(self.params[0]))) * pair_dist)+torch.tensor((X.T@X +1)**3)
            return ret.cpu().detach().numpy() if numpy else ret
        else:
            if not grad: 
                with torch.no_grad():
                    ret = torch.exp(-(1/(2*torch.square(self.params[0]))) * torch.tensor(pair_dist))+torch.tensor((X.T@Y +1)**3)
                return ret.cpu().detach().numpy() if numpy else ret
            pair_dist=torch.tensor(pair_dist)    
            ret = torch.exp(-(1/(2*torch.square(self.params[0]))) * pair_dist)+torch.tensor((X.T@Y +1)**3)
            return ret.cpu().detach().numpy() if numpy else ret
        
class PeriodicLinearKernel(Kernel):
    def __init__(self, params):
        super().__init__()
        self.params = torch.nn.ParameterList(params)
        #print(self.params)
    
    def forward(self, X, Y=None, grad=True, numpy=False):
        '''
        To compute the value using the RBF kernel, for each entrance of pair_dist if provided.
        Otherwise computes firt the pairwise distance matrix between X and X,
        or X and Y depending on if both are provided or not.

        Parameters
        -----------
        pair_dist : (n,n) numpy ndarray or torch.Tensor
            Distance between pairs x_i and y_j, if provided (default is None).
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. Needs to be provided
            if pair_dist is None. If Y is None, then the Gram matrix of X and X is 
            computed (default is None).
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            and pairwise_dist is None, then Gram matrix between X and Y is
            computed (default is None).
        grade : Boolean
            True if gradient should be computed (default is True).
        numpy : Boolean
            True if should return numpy instead of torch.Tensor (defualt is False).

        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix using the pair_dist, or between X and X, or X and Y depending
            on what is None.

        '''

        #assert X is not None, 'X must be provided when pair_dist is None.'
            #made this change here -CNB added not
        if Y is not None:
            pair_dist = self.pairwise_dist(X,Y)
        else:
            pair_dist = self.pairwise_dist(X)
        
        
        
        if Y is None: 
            if not grad: 
                with torch.no_grad():
                    ret = torch.exp(-2/torch.square(self.params[0])*torch.square(torch.sin(math.pi*torch.tensor(pair_dist)/self.params[1])))*torch.tensor(X.T@X)
                return ret.cpu().detach().numpy() if numpy else ret
            pair_dist=torch.tensor(pair_dist)    
            ret = torch.exp(-2/torch.square(self.params[0])*torch.square(torch.sin(math.pi*torch.tensor(pair_dist)/self.params[1])))*torch.tensor(X.T@X.T)
            return ret.cpu().detach().numpy() if numpy else ret
        else:
            if not grad: 
                with torch.no_grad():
                    ret = torch.exp(-2/torch.square(self.params[0])*torch.square(torch.sin(math.pi*pair_dist/self.params[1])))*torch.tensor(X.T@Y)
                return ret.cpu().detach().numpy() if numpy else ret
            pair_dist=torch.tensor(pair_dist)    
            ret = torch.exp(-2/torch.square(self.params[0])*torch.square(torch.sin(math.pi*pair_dist/self.params[1])))*torch.tensor(X.T@Y)
            return ret.cpu().detach().numpy() if numpy else ret
      
class LinearKernel(Kernel):
    def __init__(self):
        super().__init__()

    
    def forward(self, X, Y=None,grad=False, numpy=False):
        '''
        To compute the value using the Linear none paramaterized kernel.

        Parameters
        -----------
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix.
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            then Gram matrix between X and Y is computed (default is None).
        
        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix between X and X, or X and Y depending
            on whether or not Y is provided.
        '''      
        if Y is None:    
    
            return X.T @ X
   
        return X.T @ Y
       
class PeriodicRBFKernel(Kernel):
    def __init__(self, params):
        super().__init__()
        self.params = torch.nn.ParameterList(params)
        #print(self.params)
    
    def forward(self, X, Y=None, grad=True, numpy=False):
        '''
        To compute the value using the RBF kernel, for each entrance of pair_dist if provided.
        Otherwise computes firt the pairwise distance matrix between X and X,
        or X and Y depending on if both are provided or not.

        Parameters
        -----------
        pair_dist : (n,n) numpy ndarray or torch.Tensor
            Distance between pairs x_i and y_j, if provided (default is None).
        X : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. Needs to be provided
            if pair_dist is None. If Y is None, then the Gram matrix of X and X is 
            computed (default is None).
        Y : (m,m) numpy ndarray
            Matrix with points to compute Gram matrix with. If provided,
            and pairwise_dist is None, then Gram matrix between X and Y is
            computed (default is None).
        grade : Boolean
            True if gradient should be computed (default is True).
        numpy : Boolean
            True if should return numpy instead of torch.Tensor (defualt is False).

        Returns
        --------
        Gram matrix : (n,n) ndararay or torch.Tensor
            Gram matrix using the pair_dist, or between X and X, or X and Y depending
            on what is None.

        '''

        #assert X is not None, 'X must be provided when pair_dist is None.'
            #made this change here -CNB added not
        if Y is not None:
            pair_dist = self.pairwise_dist(X,Y)
        else:
            pair_dist = self.pairwise_dist(X)
        
        
        if not grad: 
            with torch.no_grad():
                pair_dist=torch.tensor(pair_dist)    
                ret1 = torch.square(self.params[0])*torch.exp(-2/torch.square(self.params[1])*torch.square(torch.sin(math.pi*pair_dist/self.params[2])))
                ret2 = torch.exp(-(1/(2*torch.square(self.params[3]))) * pair_dist)
               
                ret=ret1*ret2
                return ret.cpu().detach().numpy() if numpy else ret
        pair_dist=torch.tensor(pair_dist)    
        ret1 = torch.square(self.params[0])*torch.exp(-2/torch.square(self.params[1])*torch.square(torch.sin(math.pi*pair_dist/self.params[2])))
        ret2 = torch.exp(-(1/(2*torch.square(self.params[3]))) * pair_dist)
       
        ret=ret1*ret2
        return ret.cpu().detach().numpy() if numpy else ret
