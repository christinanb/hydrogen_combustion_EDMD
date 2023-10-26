import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.edmd import TruncEDMD, KernelEDMD, DMD
from src.data import get_nonlinear_data
import pandas as pd
import math as m
import torch
from src.training_edmd import GD_edmd_training

def OOS_tradjectory(t_eval,initial_condition,method,kernel,x,y,results):
    
    output = np.zeros((len( t_eval),2))
    if type(method).__name__ == "DMD":
        output=method.single_tradjectory(t_eval,initial_condition,results)   
    if type(method).__name__ == "TruncEDMD":
        ic=initial_condition.reshape(1,2).T
        output=method.single_tradjectory(t_eval,ic,x.T,y.T,kernel,results) 
       
    return output

def transforms(original,df_Original,initial_conditions,t_eval):
    original_trans=original
    initial_conditions_trans= np.sin(initial_conditions)
    df_trans=np.sin(df_Original)
    return df_trans,initial_conditions_trans,original_trans


def normalize(x,y,original,initial_conditions,t_eval):#
    x_norm=(x-np.mean(original,axis=0))/np.std(original,axis=0)
    y_norm=(y-np.mean(original,axis=0))/np.std(original,axis=0)
    original_norm=(original-np.mean(original,axis=0))/np.std(original,axis=0)
    initial_conditions_norm= (initial_conditions-np.mean(original,axis=0))/np.std(original,axis=0)
    
    return x_norm,y_norm,initial_conditions_norm,original_norm
         
def OOS_system_normalize(t_eval,condition,original,func):

    solution,_=func(t_eval,condition)
    solution_norm=(solution-np.mean(original,axis=0))/np.std(original,axis=0)
    initial_conditions_norm= (condition-np.mean(original,axis=0))/np.std(original,axis=0)
    
    return solution_norm,initial_conditions_norm


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def dynamicscombust(func,initial_conditions,t_end,timesteps,factor,t_begin):
    t_eval=np.linspace(0, t_end, timesteps)
    int_begin=int(t_begin//(t_end/timesteps))
    
    timesteps_fac=m.ceil((timesteps-int_begin)/factor)
    

    x = np.zeros((len(initial_conditions)*(timesteps_fac-1), 2))
    y = np.zeros((len(initial_conditions)*(timesteps_fac-1), 2))
    original= np.zeros((len(initial_conditions)*(timesteps_fac), 2))
    
    for ic, condition in enumerate(initial_conditions):
        solution,df=func(t_eval,condition)  
      
        original[ic*(timesteps_fac):ic*(timesteps_fac)+(timesteps_fac), :]=solution[int_begin::factor] 
        size=solution[int_begin::factor].shape[0]
        x[ic*(size-1) :ic*(size-1)+(size-1), :]=original[ic*(size):ic*(size)+(size)-1, :]
        y[ic*(size-1) :ic*(size-1)+(size-1), :]=original[ic*(size)+1:ic*(size)+(size), :]
 
    initial_conditions.T
    x,y,initial_conditions_norm,original_norm= normalize(x,y,original,initial_conditions,t_eval) 
   
    t_eval_frac= np.linspace(t_begin, t_end, num= timesteps_fac)
    

    return x,y,initial_conditions_norm,original_norm,original,t_eval_frac    


def dynamicshopf(func,initial_conditions,t_end,timesteps,factor):
    t_eval=np.linspace(0, t_end, timesteps)

    timesteps_fac=m.ceil(timesteps/factor)

    x = np.zeros((len(initial_conditions)*(timesteps_fac-1), 2))
    y = np.zeros((len(initial_conditions)*(timesteps_fac-1), 2))
    original= np.zeros((len(initial_conditions)*(timesteps_fac), 2))

    for ic, condition in enumerate(initial_conditions):
        solution,df=get_nonlinear_data(t_eval,condition,func)
       
        original[ic*(timesteps_fac):ic*(timesteps_fac)+(timesteps_fac), :]=solution[::factor] 
        size=solution[::factor].shape[0]
        x[ic*(size-1) :ic*(size-1)+(size-1), :]=original[ic*(size):ic*(size)+(size)-1, :]
        y[ic*(size-1) :ic*(size-1)+(size-1), :]=original[ic*(size)+1:ic*(size)+(size), :]

    initial_conditions.T
   
    t_eval= np.linspace(0, t_end, timesteps_fac)

    return x,y,original,t_eval



def traininglandscape(initial_conditions,t_end,timesteps,factor,trunc,
                      method,kernel,parameters,func):
    losstrain=np.empty(len(parameters))
   
    x,y,original,t_eval=dynamicshopf(func,initial_conditions,t_end,timesteps,factor)
 
    for p,param in enumerate(parameters):
    
        parameter=torch.tensor(param,requires_grad=True)   
   
        kernel1 = kernel([parameter])
 
        train_loss, min_loss= GD_edmd_training(1,1, x.T, y.T, kernel1, method,trunc, lr=1e-3,full=True, penalty=True)
        losstrain[p]=train_loss[0]
  
    return losstrain,min_loss

def traininglandscapenorm(initial_conditions,t_end,timesteps,factor,trunc,method,
                          kernel,parameters,func,t_begin=0):
    losstrain=np.empty(len(parameters))
   
    x,y,initial_conditions_norm,original_norm,original,t_eval_frac=dynamicscombust(func,initial_conditions,t_end,timesteps,factor,t_begin)
 
    for p,param in enumerate(parameters):
    
        parameter=torch.tensor(param,requires_grad=True)    
        
        kernel1 = kernel([parameter])
 
        train_loss, min_loss = GD_edmd_training(1,1, x.T, y.T, kernel1, method,trunc, lr=1e-3,full=True, penalty=True)
        losstrain[p]=train_loss[0]
    return losstrain,min_loss
        

def gridsearch(factors,truncs,parameters,initial_conditions,initial_conditions_sample,
               timesteps,t_end,method,kernel,func):
    #hint: plot the trajectories from each dynamic system to better visualize best paramter combintions (taken out due to too much data to visualize)
    lossperstep=np.empty([len(factors),len(truncs)])
    OOSlossperstep=np.empty([len(factors),len(truncs)])
    optim=np.empty([len(factors),len(truncs)])

    for i,factor in enumerate(factors):
        
        x,y,original,t_eval=dynamicshopf(func,initial_conditions,t_end,timesteps,factor)
         
        for p,param in enumerate(parameters):
            if param is None:
                kernel=kernel
            else:
                parameter=torch.tensor(param,requires_grad=True) 
                kernel1=kernel([parameter])
          
            for t,trunc in enumerate(truncs):
                Kernel_results=method.edmd_computations(x.T,y.T,kernel1,trunc)
                method.preloss_computation_full()
                lossperstep[i,t] = method.loss_full(kernel1, x.T, y.T, penalty=True)
                for condition in initial_conditions:
                    condition=np.array([condition],ndmin=2)
            
                #calculate out of sample trajectory
                
                sumloss=0
                for ic,condition in enumerate(initial_conditions_sample):
                    solution,df=get_nonlinear_data(t_eval,condition,func)
                    condition=np.array([condition],ndmin=2)
                    predicted=method.single_tradjectory(t_eval,condition.T,x.T,y.T,kernel1,Kernel_results) 
                    sumloss = sumloss+np.sqrt(np.sum(np.square((abs(predicted-solution))))/len(t_eval))
                    
                OOSlossperstep[i,t] =sumloss/len(initial_conditions_sample)
                optim[i,t]=np.sqrt((OOSlossperstep[i,t]**2+lossperstep[i,t]**2))
    return lossperstep,OOSlossperstep,optim
    

def gridsearchnorm(factors,truncs,parameters,initial_conditions,initial_conditions_sample,
               timesteps,t_end,method,kernel,func,t_begin=0):
    #hint: plot the trajectories from each dynamic system to better visualize best paramter combintions
    
    if parameters.ndim == 2:
        lossperstep=np.empty([parameters.shape[0],len(truncs)])
        OOSlossperstep=np.empty([parameters.shape[0],len(truncs)])
        optim=np.empty([parameters.shape[0],len(truncs)])
    
    elif parameters.size == 0:
        lossperstep=np.empty([1,len(truncs)])
        OOSlossperstep=np.empty([1,len(truncs)])
        optim=np.empty([1,len(truncs)])
        
    else:
        lossperstep=np.empty([len(parameters),len(truncs)])
        OOSlossperstep=np.empty([len(parameters),len(truncs)])
        optim=np.empty([len(parameters),len(truncs)])

    if parameters.size==0:
       kernel1=kernel()
       
       for i,factor in enumerate(factors):
            x,y,initial_conditions_norm,original_norm,original,t_eval_frac=dynamicscombust(func,initial_conditions,
            t_end,timesteps,factor,t_begin)
            for t,trunc in enumerate(truncs):
                Kernel_results=method.edmd_computations(x.T,y.T,kernel1,trunc)
                method.preloss_computation_full()
                lossperstep[0,t] = method.loss_full(kernel1, x.T, y.T, penalty=True).detach().numpy()
                for condition in initial_conditions_norm:
                    condition=np.array([condition],ndmin=2)
                  
                    Fx=method.single_tradjectory(t_eval_frac,condition.T,x.T,y.T,kernel1,Kernel_results)  
                    
                #initial_conditions_sample=np.array([[0.065,779],[0.077,781],[0.074,775]])
                sumloss=0
                for ic,condition in enumerate(initial_conditions_sample):
                    t_eval=np.linspace(0, t_end, timesteps) 
                    dynamics_norm,initial_condition_norm_sample=OOS_system_normalize(t_eval,condition,original,func)
                    int_begin=int(t_begin//(t_end/timesteps)) 
                    dynamics_norm=dynamics_norm[int_begin::factor]

                    initial_condition_norm_sample=np.array([initial_condition_norm_sample],ndmin=2)
                    predicted=method.single_tradjectory(t_eval_frac,initial_condition_norm_sample.T,x.T,y.T,kernel1,Kernel_results) 
                    sumloss = sumloss+np.sqrt(np.sum(np.square((abs(predicted-dynamics_norm))))/len(t_eval_frac))
              
                    OOSlossperstep[0,t] =sumloss/len(initial_conditions_sample)
                    if OOSlossperstep[0,t]<0.5:
                        optim[0,t]=np.sqrt((OOSlossperstep[0,t]**2+lossperstep[0,t]**2))
                    else:
                        optim[0,t]=np.nan
    
    elif parameters.ndim==2:
        
        for p in range(parameters.shape[0]):
            parameter=[torch.tensor(parameters[p,0],requires_grad=True), torch.tensor(parameters[p,1],requires_grad=True)]
            kernel1=kernel(parameter)
            for i,factor in enumerate(factors):
                x,y,initial_conditions_norm,original_norm,original,t_eval_frac=dynamicscombust(func,initial_conditions,
                t_end,timesteps,factor,t_begin)
                for t,trunc in enumerate(truncs):
                    Kernel_results=method.edmd_computations(x.T,y.T,kernel1,trunc)
                    method.preloss_computation_full()

                    lossperstep[p,t] = method.loss_full(kernel1, x.T, y.T, penalty=True).detach().numpy()

                    for condition in initial_conditions_norm:
                        condition=np.array([condition],ndmin=2)
                    
                        Fx=method.single_tradjectory(t_eval_frac,condition.T,x.T,y.T,kernel1,Kernel_results)  
                        
                    #calculate out of sample trajectory
                    #initial_conditions_sample=np.array([[0.065,779],[0.077,781],[0.074,775]])
                    sumloss=0
                    for ic,condition in enumerate(initial_conditions_sample):
                        t_eval=np.linspace(0, t_end, timesteps) 
                        dynamics_norm,initial_condition_norm_sample=OOS_system_normalize(t_eval,condition,original,func)
                        int_begin=int(t_begin//(t_end/timesteps)) 
                        dynamics_norm=dynamics_norm[int_begin::factor]

                        initial_condition_norm_sample=np.array([initial_condition_norm_sample],ndmin=2)
                        predicted=method.single_tradjectory(t_eval_frac,initial_condition_norm_sample.T,x.T,y.T,kernel1,Kernel_results) 
                        sumloss = sumloss+np.sqrt(np.sum(np.square((abs(predicted-dynamics_norm))))/len(t_eval_frac))
                    
                        OOSlossperstep[p,t] =sumloss/len(initial_conditions_sample)
                        if OOSlossperstep[p,t]<0.7:
                            optim[p,t]=np.sqrt((OOSlossperstep[p,t]**2+lossperstep[p,t]**2))
                        else:
                            optim[p,t]=np.nan

    else:
        
        for p,param in enumerate(parameters):
            parameter=torch.tensor(param,requires_grad=True) 
            kernel1=kernel([parameter])
            for i,factor in enumerate(factors):
                x,y,initial_conditions_norm,original_norm,original,t_eval_frac=dynamicscombust(func,initial_conditions,
                t_end,timesteps,factor,t_begin)
                for t,trunc in enumerate(truncs):
                    Kernel_results=method.edmd_computations(x.T,y.T,kernel1,trunc)
                    method.preloss_computation_full()

                    lossperstep[p,t] = method.loss_full(kernel1, x.T, y.T, penalty=True).detach().numpy()

                    for condition in initial_conditions_norm:
                        condition=np.array([condition],ndmin=2)
                    
                        Fx=method.single_tradjectory(t_eval_frac,condition.T,x.T,y.T,kernel1,Kernel_results)  
                        
                    #calculate out of sample trajectory
                    #initial_conditions_sample=np.array([[0.065,779],[0.077,781],[0.074,775]])
                    sumloss=0
                    for ic,condition in enumerate(initial_conditions_sample):
                        t_eval=np.linspace(0, t_end, timesteps) 
                        dynamics_norm,initial_condition_norm_sample=OOS_system_normalize(t_eval,condition,original,func)
                        int_begin=int(t_begin//(t_end/timesteps)) 
                        dynamics_norm=dynamics_norm[int_begin::factor]

                        initial_condition_norm_sample=np.array([initial_condition_norm_sample],ndmin=2)
                        predicted=method.single_tradjectory(t_eval_frac,initial_condition_norm_sample.T,x.T,y.T,kernel1,Kernel_results) 
                        sumloss = sumloss+np.sqrt(np.sum(np.square((abs(predicted-dynamics_norm))))/len(t_eval_frac))
                    
                        OOSlossperstep[p,t] =sumloss/len(initial_conditions_sample)
                        if OOSlossperstep[p,t]<0.5:
                            optim[p,t]=np.sqrt((OOSlossperstep[p,t]**2+lossperstep[p,t]**2))
                        else:
                            optim[p,t]=np.nan
        
    return lossperstep,OOSlossperstep,optim
