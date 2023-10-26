from .cbo_torch.cbo import minimize
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.kernels import RBFKernel, LinearKernel, PolynomialKernel,SigmoidKernel
from tqdm import tqdm
import time



def edmd_energy_func(x,y, kernel, edmd, full=False, penalty=False):
    '''
    Function to compute energy for CBO_edmd_training. Work in progress.
    '''
    pair_dist_xx = kernel.pairwise_dist(x)
    pair_dist_xy = kernel.pairwise_dist(x,y)

    def energy(parameters):
        kernel.params = parameters
        edmd.compute_K(pair_dist_xx, pair_dist_xy, kernel)
        
        if full:
            edmd.preloss_computation_full()
            loss = edmd.loss_full(kernel, pair_dist_xx, pair_dist_xy, x, y, penalty=penalty)
        else:    
            edmd.preloss_computation_eigen()
            loss = edmd.loss_eigen(kernel, pair_dist_xx, pair_dist_xy, x, y, penalty=penalty)
        
    
        return loss

    return energy

def CBO_edmd_training(
    x,
    y,
    kernel,
    edmd,  
    dimensionality, 
    num_particles,
    initial_distribution, 
    dt, 
    l, 
    sigma, 
    alpha, 
    anisotropic, 
    epochs, 
    batch_size,
    full=False,
    penalty=False
    ):
    '''
    Consensus based optimization. Gradient free ---- work in progress.
    '''
    energy = edmd_energy_func(x,y, kernel, edmd, full, penalty)
    return minimize(energy, dimensionality, num_particles, initial_distribution,
                    dt, l, sigma, alpha, anisotropic, batch_size=batch_size, epochs=epochs)


def GD_edmd_training(num_epoch, grad_steps, x,y, kernel, edmd, trunc,lr=1e-4, full=True, penalty=False):
    '''
    Running iterations of computing the Koopman approximation, holding the kernel parameters fixed.
    Then updating the kernel parrameters holding the Koopman approximation fixed, using gradient descent
    to update parameters. 

    Parameters
    -----------
    num_epoch : positive int
        Number of times to runn the iterative update scheme.
    grad_steps : positive int
        Number of gradient descent steps for each epoch.
    x : (N,D) ndarray
        Training snapshots from a dynamical system with evolution operator T. 
    y : (N,D) ndarray
        Training snapshots from applying T to x, T(x_i) = y_i
    kernel : subclasss of Kernel
        Kernel with the parameters optimized over.
    edmd : subclass of EDMD
        Object that computes the Koopman approximation, loss, Koopman modes etc.
    lr  : float
        Learning rate to the gradient dedscent step (default 1e-4).
    full : Boolean
        True if the full loss is to be used, instead of eigen loss (default True)
    penalty : Boolean
        Adding the penalty term to the loss function, to for instance avoid trivial solutions 
        (default False).

    Return
    -------
    loss : tourch.Double tensor
        Final loss.  
    '''

    train_losses = []

    optimizer=torch.optim.RMSprop(kernel.params, lr=lr,momentum=0.9)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    #optimizer = torch.optim.SGD(kernel.parameters(), lr)
    #optimizer = torch.optim.Adam(kernel.parameters(), lr)
    #optimizer=torch.optim.ASGD(kernel.parameters(), lr)
    
    min_loss = np.inf
    train_losses = []
    min_param=torch.tensor(0.0, requires_grad=True)
 
    for epoch in range(num_epoch):
        #training
       
        
        edmd.compute_K(x, y, kernel,trunc)
  
        #edmd.compute_K(x, y, kernel)

    #this is where the koopman operator is approximated
        if full:
            edmd.preloss_computation_full()
        else:
            edmd.preloss_computation_eigen()

        for ite in range(grad_steps):
            optimizer.zero_grad() #reset the gradients here

            if full:
                loss = edmd.loss_full(kernel, x, y, penalty=penalty)
              
            else:
                loss = edmd.loss_eigen(kernel, x, y, penalty=penalty)
       
            loss.backward()
            
            
            optimizer.step()
            #scheduler.step()
        train_losses.append(loss.detach().numpy())
        if loss < min_loss:
            min_loss=loss
            min_param=kernel.parameters()

             
    return train_losses,min_loss

    
