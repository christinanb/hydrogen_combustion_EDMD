import torch
import scipy.linalg
import scipy.spatial.distance as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from src.kernels import RBFKernel, LinearKernel
device = torch.device("cuda:0")

class EDMD:
    def __init__(self):
        None
        
    def compute_K(self):
        '''
        Computes the Koopman approximation.

        Raises
        ------
        NotImplementedError
            Needs to be implemented in subclasses.
        '''
        raise NotImplementedError("Subclasses should implement this!")
        
    def training(self):
        '''
        Trains the parameters of the kernel.

        Raises
        ------
        NotImplementedError
            Needs to be implemented in subclasses.
        '''
        raise NotImplementedError("Subclasses should implement this!")
        
    def compute_eigen(self, left=True, right=True):
        '''
        Computes eigenvalues and vectors for Kooopman approximation K, where K is an MxM matrix. 
        K is the matrix corresponding to last time compute_K ran.

        Parameters
        ----------
        left : Boolean
            If true, left eigenvectors are computed (default is True).
        right: Boolean
            If true, right eigenvectors are computed (default is True).
        
        Returns
        -------
        lam : (N,) or (2,N) double or complex ndarray
            Eigenvalues corresponding to the K.
        w : (N,N) double or complex ndarray.
            Normalized left eigenvectors corresponding to eigenvalue lam[i] is the 
            row w[i, :]. Only returned if left=True.
        v : (N,N) double or complex ndarray.
            Normalized left eigenvectors corresponding to eigenvalue lam[i] is the 
            column v[:, i]. Only returned if right=True.
        '''
       # if torch.is_tensor(self.K):
           # K_nump = self.K.cpu().detach().numpy()             
       # else:
           # K_nump = self.K
            
        #print("K",self.K.shape)
        lam, w_tran, v = scipy.linalg.eig(self.K, left=True, right=True)
        w = w_tran.conj().T
        
        if left and right: return lam, w, v
        if left: return lam, w
        if right: return lam, v

    def vis(self,t):
        '''
        Trains the parameters of the kernel.

        Raises
        ------
        NotImplementedError
            Needs to be implemented in subclasses.
        '''
        raise NotImplementedError("Subclasses should implement this!")


class KernelEDMD(EDMD):
    def __init__(self, regularizer=1e-3):
        super().__init__()
        self.regularizer = regularizer

    def compute_K(self, pair_dist_xx, pair_dist_xy, kernel):
        '''
        Computes the Koopman approximation for a given X, Y, and kernel.
        The dictionary is explicitly defined, and psi_i(z) = kernel(z, x_i), where (x_i)_{i=1}^N are the training points. 
        No gradient is stored when computing K.

        Parameters
        ----------
        pair_dist_xx : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and x_j.
        pair_dist_xy : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and y_j.
        kernel : Subclass of Kernel
                The kernel that uses the pairwise distances, and calculates the kernel value to produce 
                gram matrices.
        
        Returns
        -------
        K : (N,N) double or complex ndarray
            Koopman approximation K.
        '''
        with torch.no_grad():
            self.gram_x = kernel(pair_dist_xx, grad = False, numpy = True)
            self.gram_y = kernel(pair_dist_xy, grad = False, numpy = True)
          
            self.K = (scipy.linalg.lstsq(self.gram_x.T, self.gram_y.T, cond=self.regularizer)[0]).T

        return self.K

    def compute_koop_modes(self, target, x, y, kernel):
        '''
        Computes the Koopman modes for a specific function f:F^D -> F^d evaluated at y, where
        F is either the real or complex field.

        Parameters
        ----------
        target : (d, n) double or complex ndarray
            Matrix where each column is the specific function evaluated at y_i, i.e. f(y_i). 
            d is the number of outputs of f.
        x : (N, D) double or complex ndarray
            Matrix with N training snapshots that was used to compute koopman approximation K.
        y : (n,D) double or complex ndarray
            Matrix with n snapshots that f is evaluated at.
        kernel : subclasss of Kernel

        Returns
        -------
        modes : (d,N) or (,N) ndarray
            The approximated Koopman modes. 
        '''
        self.compute_phi(x, y, kernel)
        identity_mat = np.eye((self.phi.shape[0]))

        return (scipy.linalg.lstsq(self.phi.T, target.T, cond=self.regularizer)[0]).T

    def compute_phi(self, x=None, y=None, kernel=None):
        '''
        Computes the approximated eigenfunctions based of the computed Koopman approximation K,
        which is the matrix last computed by compute_K.

        Parameters
        ----------
        x : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K. 
            Only needed if one wish to evaluate eigenfunctions at y (default is None).
        y : (n,D) ndarray
            Data used to evaluate the eigenfunctions at, optional (default is None).
        kernel : subclasss of Kernel
            Kernel that are used to explicitly define the dictionary. Only needed if one wish 
            to evaluate eigenfunctions at y (default is None).

        Returns
        -------
        phi : (N,N) double or complex ndarray
            Approximation of eigenfunctions of Koopman approximation K.
        phi_y : (N,N) ndarray
            Eigenfunctions evaluated at y. (Only returned if y is not None)
        
        Raises
        ------
        ValueError
            If not kernel, y, and x are all None xor not None.
        '''
        lam, w = self.compute_eigen(right=False)
        lam, self.w = np.diag(lam), w

        self.phi_pre = self.w  

        if y is None and kernel is None and x is None:
            return self.phi_pre
        
        if y is not None and kernel is not None and  x is not None:
            gram_y = kernel(X=x, Y=y, numpy=True, grad=False)
            self.phi = self.phi_pre @ gram_y
            return self.phi, self.phi_pre
                                   
        raise ValueError("Either 'y', 'x', and 'kernel' must be None, or all be not None.")
                

    def preloss_computation_full(self):
        '''
        Computes necessary computations needed when computing the full loss function,
        components are not dependent on the parameters we optimize for in the loss function.

        Returns
        --------
        K : (N,N) torch.Tensor
            The Koopman approximation computed in compute_K, only as a torch tensor.
        '''
        with torch.no_grad():
            self.K = torch.tensor(self.K)
        return self.K

    def preloss_computation_eigen(self):
        '''
        Computes necessary computations needed when computing eigen loss function,
        components are not dependent on the parameters we optimize for in the loss function.

        Returns
        --------
        lam : (N,) or (2,N) torch.Tesor
            Eigenvalues corresponding to the K.
        w : (N,N) torch.Tensor
            Normalized left eigenvectors corresponding to eigenvalue w[i] is the 
            row w[i, :].
        '''
        with torch.no_grad():
            lam, w = self.compute_eigen(right=False)
            self.lam, self.w = torch.tensor(np.diag(lam)), torch.tensor(w)
        return self.lam, self.w

    def loss_full(self, kernel, pair_dist_xx, pair_dist_xy, x=None, y=None, penalty=False):
        '''
        Computes the full loss, recording the gradient so that auto diff can be used later.
        
        Parameters
        -----------
        kernel : subclasss of Kernel
            Kernel with the parameters optimized over.
        pair_dist_xx : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and x_j.
        pair_dist_xy : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and y_j.
        x : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K. 
            Only needed if one wish to add penalty term (default is None).
        y : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K
            Only needed if one wish to add penalty term (default is None).
        penalty : Boolean
            True only if penaalty term is to be added. Idetity function of Y minus the 
            predicted values by the Koopman approximation (default is False).

        Returns
        --------
        loss : torch.DoubleTensor
            Loss computed given Koopman approximation from compute_K and data/snapshots.
        '''
        gram_x = kernel(torch.tensor(pair_dist_xx))
        gram_y = kernel(torch.tensor(pair_dist_xy))

        diff = gram_y - self.K @ gram_x

        if penalty:
            assert not(x is None or y is None), "Both x and y must be not None if penalty is added"
            koop_modes = self.compute_koop_modes(y.T, x, y, kernel)
            
            gram_y = gram_y.type(torch.complex128)
            eig_func = torch.tensor(koop_modes @ self.phi_pre).type(torch.complex128)
            pen_term = torch.tensor(y).T - (eig_func @ gram_y).type(torch.double)
            
            return torch.square(torch.norm(diff)) + torch.square(torch.norm(pen_term))
        
        return torch.square(torch.norm(diff))

    def loss_eigen(self, kernel, pair_dist_xx, pair_dist_xy, x=None, y=None, penalty=False):
        '''
        Computes the eigen loss, recording the gradient so that auto diff can be used later.
        The loss uses the approximated eigenfunctions based off compute_K, with
        eigenfunctions of y - lambda * eigenfunctions of x.
        
        Parameters
        -----------
        kernel : subclasss of Kernel
            Kernel with the parameters optimized over.
        pair_dist_xx : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and x_j.
        pair_dist_xy : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and y_j.
        x : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K. 
            Only needed if one wish to add penalty term (default is None).
        y : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K
            Only needed if one wish to add penalty term (default is None).
        penalty : Boolean
            True only if penaalty term is to be added. Idetity function of Y minus the 
            predicted values by the Koopman approximation (default is False).

        Returns
        --------
        loss : torch.DoubleTensor
            Loss computed given Koopman approximation from compute_K and data/snapshots.
        '''
        gram_x = kernel(torch.tensor(pair_dist_xx)).type(torch.complex128)
        gram_y = kernel(torch.tensor(pair_dist_xy)).type(torch.complex128)

        rs = self.lam.type(torch.complex128) @ self.w.type(torch.complex128) @ gram_x
        diff = self.w.type(torch.complex128) @ gram_y - rs
        
        if penalty:
            assert not(x is None or y is None), "Both x and y must be not None if penalty is added"
            koop_modes = self.compute_koop_modes(y.T, x, y, kernel)

            gram_y = gram_y.type(torch.complex128)
            eig_func = torch.tensor(koop_modes @ self.phi_pre).type(torch.complex128)
            pen_term = torch.tensor(y).T - (eig_func @ gram_y).type(torch.double)

            return torch.square(torch.norm(diff)) + torch.square(torch.norm(pen_term))
        return torch.square(torch.norm(diff))

    def to_trunc_edmd(self, x, y, kernel):
        '''
        Parameters
        -----------
        x : (N,D) ndarray
            Training snapshots from a dynamical system with evolution operator T. 
        y : (N,D) ndarray
            Training snapshots from applying T to x, T(x_i) = y_i
        kernel : subclasss of Kernel
            Kernel used to compute the Koopman approximation.
        
        Returns
        --------
        lam : (N,) or (2,N) torch.Tesor
            Eigenvalues corresponding to the spectrum of truncated SVD Koopman approximaton,
            given the Koopman approximation computed in compute_K.
        w_trunc : (N,N) double or complex ndarray.
            Normalized left eigenvectors corresponding to eigenvalue lam[i] is the 
            row w[i, :].
        v_trunc : (N,N) double or complex ndarray.
            Normalized left eigenvectors corresponding to eigenvalue lam[i] is the 
            column v[:, i]. 
        '''
        pair_dist_xx = kernel.pairwise_dist(x)
        pair_dist_xy = kernel.pairwise_dist(x,y)
        gram_x = kernel(pair_dist_xx)


        self.compute_K(pair_dist_xx, pair_dist_xy, kernel)
        sigma_sq, z = scipy.linalg.eigh(gram_x)
        sigma = np.diag(np.sqrt(sigma_sq))
        sigma_inv = np.linalg.pinv(sigma)


        #K_trunc = (z @ sigma_inv).T @ self.K @ (z @ sigma)
        lam, w, v = self.compute_eigen()
        lam = np.diag(lam)

        w_trunc = w @ sigma_inv @ z.T
        v_trunc = z @ sigma @ v

        return lam, w_trunc, v_trunc





class TruncEDMD(EDMD):
    def __init__(self):
        super().__init__()
    
    def compute_K(self,x, y, kernel,trunc=10):
        '''
        Computes the Koopman approximation for a given X, Y, and kernel, where the
        dictionary is implicitly defined through the kernel, and then using the Truncated SVD to 
        compute the approximation.
        No gradient is stored when computing K.

        Parameters
        ----------
        pair_dist_xx : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and x_j.
        pair_dist_xy : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and y_j.
        kernel : Subclass of Kernel
                The kernel that uses the pairwise distances, and calculates the kernel value to produce 
                gram matrices.
        
        Returns
        -------
        K : (N,N) double or complex ndarray
            Koopman approximation K.
        '''
        with torch.no_grad():
          
            self.gram_x = kernel(X=x, Y=None,grad = False, numpy = True) 
           
            self.gram_y = kernel(X=x, Y=y, grad = False, numpy = True)
           
            sigma_sq, z = scipy.linalg.eigh(self.gram_x)
            
          

            sigma_sq = sigma_sq * (sigma_sq >1e-30)#-6 #5e-3 for all together 7e-7
           
            
            top_n_indices = np.argsort(sigma_sq)[-trunc:]
            mask = np.zeros_like(sigma_sq, dtype=bool)
            mask[top_n_indices] = True
            sigma_sq[~mask] = 0
            sigma = np.diag(np.sqrt(sigma_sq))
         

            #print("trunc value ",trunc,"num of non zero rows ",np.count_nonzero(sigma))
            
            sigma_inv = np.linalg.pinv(sigma,hermitian=True) #might have to fix hermitian here
            p = z@sigma_inv
           
            self.K = p.T @ self.gram_y @ p 
            self.K = self.K
          

            self.Z = z
            self.Sigma = sigma
            self.Sigma_inv = sigma_inv

        return self.K, self.Z, self.Sigma_inv, self.Sigma

    def compute_koop_modes(self, target, x, y, kernel, regularizer=1e-20):#20
        '''
        Computes the Koopman modes for a specific function f:F^D -> F^d evaluated at y, where
        F is either the real or complex field.

        Parameters
        ----------
        target : (d, n) double or complex ndarray
            Matrix where each column is the specific function evaluated at y_i, i.e. f(y_i). 
            d is the number of outputs of f.
        x : (N, D) double or complex ndarray
            Matrix with N training snapshots that was used to compute koopman approximation K.
        y : (n,D) double or complex ndarray
            Matrix with n snapshots that f is evaluated at.
        kernel : subclasss of Kernel

        Returns
        -------
        modes : (d,N) or (,N) ndarray
            The approximated Koopman modes. 
        '''
        self.compute_phi(x, y, kernel)
        return (scipy.linalg.lstsq(self.phi.T, target.T, cond=regularizer,overwrite_a=True,overwrite_b=True)[0]).T #lapack_driver="gelsy"
        
    def compute_phi(self, x, y, kernel=None):
        '''
        Computes the approximated eigenfunctions based of the computed Koopman approximation K,
        which is the matrix last computed by compute_K.

        Parameters
        ----------
        x : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K. 
            Only needed if one wish to evaluate eigenfunctions at y (default is None).
        y : (n,D) ndarray
            Data used to evaluate the eigenfunctions at, optional (default is None).
        kernel : subclasss of Kernel
            Kernel used to approximate Koopman operator.
            Only needed if one wish to evaluate eigenfunctions at y (default is None).

        Returns
        -------
        phi : (N,N) double or complex ndarray
            Approximation of eigenfunctions of Koopman approximation K.
        phi_y : (N,N) ndarray
            Eigenfunctions evaluated at y. (Only returned if y is not None)

        Raises
        ------
        ValueError
            If not kernel, y, and x are all None xor not None.
        '''
        lam, w = self.compute_eigen(right=False)
        
        self.lam, self.w = np.diag(lam), w
    
        self.phi_pre = self.w @ self.Sigma_inv @ self.Z.T 

        if y is None and kernel is None and x is None:
            return self.phi_pre
        
        if y is not None and kernel is not None and x is not None:
            gram_y = kernel(x, y, numpy=True, grad=False)
            self.phi = self.phi_pre @ gram_y
            return self.phi, self.phi_pre
                                   
        raise ValueError("Either 'y', 'x', and 'kernel' must be None, or all be not None.")
                
    
    def preloss_computation_full(self):
        '''
        Computes necessary computations needed when computing the full loss function,
        components are not dependent on the parameters we optimize for in the loss function.

        Returns
        --------
        ls : (N,N) torch.Tensor
            "Left hand side" of the loss function, to be multiplied with the gram matrix of kernel(x,y).
        rs : (N,N) torch.Tensor
            "Right hand side" of the loss function, to be substracted from the left hand side.
        '''
        with torch.no_grad():
            lam, w, v = self.compute_eigen()
            self.lam=lam
            lam = np.diag(lam)
            ls = v @ w @ self.Sigma_inv @ self.Z.T
            rs = v @ lam @ w @ self.Sigma @ self.Z.T
            
            self.ls = torch.tensor(np.real(ls))
            self.rs = torch.tensor(np.real(rs))
            self.lam = lam
            self.w = w
            self.v = v

        return self.ls, self.rs
    
    def preloss_computation_eigen(self):
        '''
        Computes necessary computations needed when computing eigen loss function,
        components are not dependent on the parameters we optimize for in the loss function.

        Returns
        --------
        ls : (N,N) torch.Tensor Complex
            "Left hand side" of the loss function, to be multiplied with the gram matrix of kernel(x,y).
        rs : (N,N) torch.Tensor Complex
            "Right hand side" of the loss function, to be substracted from the left hand side.
        '''
        with torch.no_grad():
            self.lam, w = self.compute_eigen(right=False)
            self.lam, self.w = np.diag(self.lami), w

            self.ls = self.w @ self.Sigma_inv @ self.Z.T
            self.rs = self.lam @ self.w @ self.Sigma @ self.Z.T

            self.ls = torch.tensor(self.ls, dtype=torch.complex128)
            self.rs = torch.tensor(self.rs, dtype=torch.complex128)
            
        return self.ls, self.rs
            
    
    def loss_full(self, kernel, x, y, penalty=False):
        '''
        Computes the full loss, recording the gradient so that auto diff can be used later.
        
        Parameters
        -----------
        kernel : subclasss of Kernel
            Kernel with the parameters optimized over.
        pair_dist_xx : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and x_j.
        pair_dist_xy : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and y_j.
        x : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K. 
            Only needed if one wish to add penalty term (default is None).
        y : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K
            Only needed if one wish to add penalty term (default is None).
        penalty : Boolean
            True only if penaalty term is to be added. Identity function of Y minus the 
            predicted values by the Koopman approximation (default is False).

        Returns
        --------
        loss : torch.DoubleTensor
            Loss computed given Koopman approximation from compute_K and data/snapshots.
        '''
        assert not(x is None or y is None), "Both x and y must be not None if penalty is added"
        gram_y = kernel(torch.tensor(x),torch.tensor(y))
        #gram_y = kernel(x,y)
        diff = self.ls @ gram_y - self.rs

        if penalty:
            assert not(x is None or y is None), "Both x and y must be not None if penalty is added"
            koop_modes = self.compute_koop_modes(y, x, y, kernel) # this might not be correct here
            
            gram_y = gram_y.type(torch.complex128)
            eig_func = torch.tensor(koop_modes @ self.phi_pre).type(torch.complex128) 
            pen_term = torch.tensor(y) - (eig_func @ gram_y).type(torch.double)
            
            return torch.square(torch.norm(diff)) + torch.square(torch.norm(pen_term))
        return torch.square(torch.norm(diff))
    
    def loss_eigen(self, kernel, pair_dist_xx, pair_dist_xy, x=None, y=None, penalty=False):
        '''
        Computes the eigen loss, recording the gradient so that auto diff can be used later.
        The loss uses the approximated eigenfunctions based off compute_K, with
        eigenfunctions of y - lambda * eigenfunctions of x.
        
        Parameters
        -----------
        kernel : subclasss of Kernel
            Kernel with the parameters optimized over.
        pair_dist_xx : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and x_j.
        pair_dist_xy : (N,N) double or complex ndarray
            Pairwise distance between training examples x_i and y_j.
        x : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K. 
            Only needed if one wish to add penalty term (default is None).
        y : (N,D) ndarray
            Training snapshots used to approximate Koopman operator K in compute_K
            Only needed if one wish to add penalty term (default is None).
        penalty : Boolean
            True only if penaalty term is to be added. Idetity function of Y minus the 
            predicted values by the Koopman approximation (default is False).

        Returns
        --------
        loss : torch.DoubleTensor
            Loss computed given Koopman approximation from compute_K and data/snapshots.
        '''
        gram_y = kernel(X=x, Y=y)
        diff = self.ls @ gram_y.type(torch.complex128) - self.rs 

        if penalty:
            assert not(x is None or y is None), "Both x and y must be not None if penalty is added"
            koop_modes = self.compute_koop_modes(y, x, y, kernel)
            
            gram_y = gram_y.type(torch.complex128)
            eig_func = torch.tensor(koop_modes @ self.phi_pre).type(torch.complex128) 
            pen_term = torch.tensor(y) - (eig_func @ gram_y).type(torch.double)

            return torch.square(torch.norm(diff)) + torch.square(torch.norm(pen_term))

        return torch.square(torch.norm(diff))
                
    def to_kernel_edmd(self, x, y, kernel):
        '''
        Parameters
        -----------
        x : (N,D) ndarray
            Training snapshots from a dynamical system with evolution operator T. 
        y : (N,D) ndarray
            Training snapshots from applying T to x, T(x_i) = y_i
        kernel : subclasss of Kernel
            Kernel used to compute the Koopman approximation.
        
        Returns
        --------
        lam : (N,) or (2,N) torch.Tesor
            Eigenvalues corresponding to the spectrum of explicit Kernel Koopman approximaton,
            given the Koopman approximation computed in compute_K.
        w_trunc : (N,N) double or complex ndarray.
            Normalized left eigenvectors corresponding to eigenvalue lam[i] is the 
            row w[i, :].
        v_trunc : (N,N) double or complex ndarray.
            Normalized left eigenvectors corresponding to eigenvalue lam[i] is the 
            column v[:, i]. 
        '''
        pair_dist_xx = kernel.pairwise_dist(x)
        pair_dist_xy = kernel.pairwise_dist(x,y)
        
        self.compute_K(pair_dist_xx, pair_dist_xy, kernel)
        
        lam, w, v = self.compute_eigen()
        lam = np.diag(lam)
        
        w_kernel = w @ (self.Z @ self.Sigma_inv).T
        v_kernel = (self.Z @ self.Sigma) @ v
        
        return lam, w_kernel, v_kernel


    def edmd_computations(self,x,y,kernel,trunc=10):    
        self.compute_K(x, y,kernel,trunc)
        self.koop_modes=self.compute_koop_modes(x, x, x, kernel)
        return self.lam,self.phi_pre,self.koop_modes
        

        
    def single_tradjectory(self, t_eval_sample, ic, x, y, kernel, results=None):
        output = np.zeros((len(t_eval_sample),x.shape[0]))
        output[0,:] = ic.reshape(1,x.shape[0])

        if results is not None:

            lam=np.array(results[0])
            phi_pre=np.array(results[1])
            koop_modes=np.array(results[2])
            
        else:    
            lam=self.lam
            
            koop_modes=self.koop_modes
            phi_pre=self.phi_pre

        gram_ic = kernel(x, ic, numpy=True, grad=False)
        psi_x0 =  phi_pre @ gram_ic
   
                  
        for index,t in enumerate(t_eval_sample[:-1]):
            #if index>=1 and index<=len(t_eval_sample): 
            Fx = koop_modes @ lam @ psi_x0
            output[index+1,:] = Fx.reshape((1,x.shape[0]))        
            gram_ic = kernel(x, np.real(Fx), numpy=True, grad=False)
                
            psi_x0 = phi_pre @ gram_ic               
         
        return output 
    

class DMD:
    def __init__(self):
        None

    def compute_eigen(self,K,left=True, right=True):
            
     
        lam, w, v = scipy.linalg.eig(K, left=True, right=True)
        
        
        if left and right: return lam, w, v
        if left: return lam, w
        if right: return lam, v  

    def compute_K_DMD(self,x,y):
     
        self.u,sigma, self.v_T = scipy.linalg.svd(x,full_matrices = False)  
        self.sigma_inv=np.diag(np.reciprocal(sigma))
       
        r=self.sigma_inv.shape[0]      
        self.u=self.u[:,:r]  
        self.v_T=self.v_T[:r,:]   

        self.K_tilde=self.u.T@y@self.v_T.T@self.sigma_inv

        lam,_,self.w=self.compute_eigen(self.K_tilde,left=True, right=True) 
        self.lam = np.diag(lam)

        self.psi=y@self.v_T.conj().T@self.sigma_inv@self.w
        self.K=self.psi@self.lam@np.linalg.pinv(self.psi)
    
    
        return self.K
    
    def single_tradjectory(self,t_eval_sample,ic,K_input=None):
        output = np.zeros((len(t_eval_sample),2))
        output[0] = ic.reshape(1,2)
        ic=ic.T
        if K_input is not None:
            K=K_input
        else :
            K=self.K
        

        for index,t in enumerate(t_eval_sample):
            if index>=1 and index<len(t_eval_sample):
                Fx=K@ic                 
                output[index] = Fx.reshape(1,2)
                ic=Fx
        
        return output  

 
