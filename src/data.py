import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import math as m
import cantera as ct
import torch
from src.training_edmd import GD_edmd_training
from src.kernels import RBFKernel


def get_linear_data(t_eval, initial_condition, dim,A):
    #t_eval = np.linspace(0, t_end, timesteps)
    solution = np.empty((len(t_eval), dim))
    expA = scipy.linalg.expm(A)
    for ti in range(len(t_eval)):
        solution[ti, :] = scipy.linalg.fractional_matrix_power(expA, ti/len(t_eval)) @ initial_condition    
    df = pd.DataFrame(data=solution,index=t_eval,columns=["x1", "x2"]) 
    return solution,df, A,expA

def get_quad_data(t, num_init):
    X, Y, A = get_linear_data(t, num_init,dim=2)
    X[:,0] = X[:,0] - np.square(X[:,1])
    X[:,1] = -np.square(X[:,0]) + X[:,1] + 2 * X[:,0] * np.square(X[:,1]) - np.power(X[:,1],4)

    Y[:,0] = Y[:,0] - np.square(Y[:,1])
    Y[:,1] = -np.square(Y[:,0]) + Y[:,1] + 2 * Y[:,0] * np.square(Y[:,1]) - np.power(Y[:,1],4)

    return X, Y, A 

def duffing_oscilator(t=None,y=None):
        a,b,d=1,-1,0.5 
        dydt=np.zeros(2)
    
        dydt[0]=y[1]
        dydt[1]=-d*y[1]-y[0]*(b+a*y[0]**2)
        return dydt


def hopf(t=None,y=None):
        mu=1
        dydt=np.zeros(2)
    
        dydt[0]=-y[1]+y[0]*(mu-y[0]**2-y[1]**2)
        dydt[1]=y[0]+y[1]*(mu-y[0]**2-y[1]**2)
        return dydt

def get_nonlinear_data(t_eval,initial_conditions,equation):
    #t_eval = np.linspace(0, t_end, timesteps)
    sol = solve_ivp(equation,(t_eval[0], t_eval[-1]), y0=initial_conditions, t_eval=t_eval)    
    #sol = solve_ivp(equation,(t_eval[0], t_eval[-1]), y0=initial_conditions)  
    df = pd.DataFrame(data=sol["y"].T,index=sol["t"],columns=["x1", "x2"],) 
    #df = pd.DataFrame(data=sol["y"].T,index=sol["t"],columns=["x1", "x2"],)
    solution=np.array(sol["y"]).T
    return solution,df



def H2_combustion1(t_eval,ic)  : 
    gas = ct.Solution('h2o2.yaml')
   
    
    p = 10.0*133.3
    
    gas.TPX = ic[1], p, {'H2':ic[0],'O2':0.12,'N2':(1.0-0.12-ic[0])}
 
    
    upstream = ct.Reservoir(gas)
    cstr = ct.IdealGasReactor(gas)
    cstr.volume = 9.0*1.0e-4
    
    env = ct.Reservoir(gas)
    w = ct.Wall(cstr, env, A=2.0, U=5.0)
    

    sccm = 1.25
    
    mdot = 2.5032937555891818e-08
    vdot =2.5032937555891818e-08/gas.density
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot)

    downstream = ct.Reservoir(gas)
    v = ct.Valve(cstr, downstream, K=0.01) 
    
    restime=cstr.volume/ vdot
   
    network = ct.ReactorNet([cstr])
   
    states = ct.SolutionArray(gas, extra=['t'])
    for t in t_eval:
        network.advance(t)
        states.append(cstr.thermo.state, t=t)

    aliases = {'H2': 'H$_2$', 'O2': 'O$_2$', 'H2O': 'H$_2$O'}
    
    for name, alias in aliases.items():
        gas.add_species_alias(name, alias)
 
    temp=np.array([states.T]).reshape(len(t_eval))
    conc=np.array([states('H2').X]).reshape(len(t_eval))
    sol = np.array([[conc], [temp]]).reshape(2,len(t_eval))
 
    solution=sol.T
   
    df = pd.DataFrame(data=np.array([sol[0],sol[1]]).T,index=t_eval,columns=["x1","x2"],) 
    return solution,df

def H2_combustion2(t_eval,ic)  : 
    gas = ct.Solution('h2o2.yaml')

    p = 9.0*133.3

    gas.TPX = ic[1], p, {'H2':ic[0],'O2':0.14,'N2':(1.0-0.14-ic[0])}
 
    
    upstream = ct.Reservoir(gas)
    cstr = ct.IdealGasConstPressureReactor(gas)
    cstr.volume =5.0e-6#-4
    
    env = ct.Reservoir(gas)
    w = ct.Wall(cstr, env, A=2.0, U=0.035)#0.02)
    

    mdot = 70.5032937555891818e-08#gas.density * vdot  # kg/s
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot)

    downstream = ct.Reservoir(gas)
    pressure_regulator = ct.MassFlowController(cstr, downstream,  mdot=mdot)

    network = ct.ReactorNet([cstr])
   
    states = ct.SolutionArray(gas, extra=['t'])
    for t in t_eval:
        network.advance(t)
     
        states.append(cstr.thermo.state, t=t)

    aliases = {'H2': 'H$_2$', 'O2': 'O$_2$', 'H2O': 'H$_2$O'}
    
    for name, alias in aliases.items():
        gas.add_species_alias(name, alias)
   
    temp=np.array([states.T]).reshape(len(t_eval))
    conc=np.array([states('H2').X]).reshape(len(t_eval))
    sol = np.array([[conc], [temp]]).reshape(2,len(t_eval))
   
    solution=sol.T

    df = pd.DataFrame(data=np.array([sol[0],sol[1]]).T,index=t_eval,columns=["x1","x2"],) 
    return solution,df





