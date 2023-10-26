# hydrogen_combustion_EDMD

This repository contains the work from my master's Thesis Applications of Kernel EDMD, where Extended Dynamic Mode decomposition is used to model the dynamics of hydrogen combustion in a CSTR. Supervisors: Dr. Felix Deitrich and Erik Bolanger (M.sc.)
# Overview

The goal of this project is to use the EDMD method to model hydrogen combustion data and predict the dynamics with starting conditions outside of those used to form the original dataset. The capabilities of this algorithm as in any machine learning application are extremely sensitive to the hyperparameters and the choice of functions in the EDMD function library. The EDMD function library is represented by kernel functions which can contain free parameters that define the type of functions in the library.  These hyperparameters are the number of eigenfunctions retained after EDMD, the kernel parameters (which indirectly relate to the functions in the function library), and the number of data points used in the algorithm. 

To find the proper parameter settings a grid search capability is integrated to perform a large-grained search of possible parameter combinations. Once that has been performed, the kernel parameters can be refined even more precisely using a gradient descent training method with a loss function that calculates the difference between the true data values and the predicted values from the EDMD method. The results of the thesis work show that because the loss landscape is full of many local minimums, the method does not always converge to an optimal parameter. Additionally, it is extremely easy to overfit the data, so it is critical to implement some sort of cross-validation; however traditional cross-validation approaches are not appropriate due to the time dependency of the data. This requires an approach where the accuracy of predicted dynamic trajectories using initial conditions outside of the original data range are evaluated (these new dynamic trajectories are called "Out of sample data"). An optimization of this value as well as an optimization of the accuracy of the in-sample predictions is required to adequately model the data. The Out-of-sample data accuracy measurement is implemented into the grid search, but not implemented into the kernel learning algorithm. 

It was found that when using optimized hyperparameters, the EDMD method with all kernels used to model the hydrogen combustion dynamics produces better out-of-sample predictions when only the dynamics resulting from changes in initial reactor temperature are modeled. Modeling the effects on the dynamics of the system  of changing both the initial reactor temperature and initial reactor hydrogen concentration did not produce as accurate of predictions. This effect is most likely due to the complicated effects of modeling the high-dimensionality system by only two dimensions and the choice of functions in the library. 

# Details
The Cantera Library  is used to model hydrogen combustion in a CSTR reactor where the concentration of hydrogen and the temperature of the reactor is measured over time. Additionally, a less complex dynamical system representing a hopf bifurcation is available as a test of concept. The data from each dynamic system is generated by solving a system of governing equations using a classical Runge Kutta solver. The data is organized into a time-series format where EDMD is performed using different kernels that represent different function libraries. The kernels available are linear, polynomial, Radial Basis, Sigmoid, and kernels which contain a combination of those. Different Jupyter notebooks are used for different test cases. 

-linearhopf.ipynb  performs EDMD on the hopf bifurcation dynamics using a linear kernel (produces the same results as classical DMD) with no kernel parameters
-polynomialhopf.ipynb performs EDMD on the hopf bifurcation dynamics using the polynomial kernel with no kernel parameters
-RBFhopf.ipynb performs EDMD on the hopf bifurcation dynamics using the Radial Basis function Kernel
-RBFreaction.ipynb performs EDMD on hydrogen combustion dynamics using the Radial Basis Function Kernel. In this case, the time series training data includes initial conditions varying in both temperature and starting Hydrogen concentration.
-RBFreactiontempalternative.ipynb performs EDMD on hydrogen combustion dynamics using hte 
-KernelComparisonReaction.ipynb performs EDMD on the hydrogen combustion dynamics using all kernels as a method of comparison. The optimal hyperparameters are found using a grid search (no parameter learning)

-- The src folder contains the code for the EDMD method, data generation, defining the kernels, and the kernel parameter training.

# References
David G. Goodwin, Harry K. Moffat, Ingmar Schoegl, Raymond L. Speth, and Bryan W. Weber. Cantera: An object-oriented software toolkit for chemical kinetics, thermodynamics, and transport processes. https://www.cantera.org, 2023. Version 3.0.0. doi:10.5281/zenodo.8137090
