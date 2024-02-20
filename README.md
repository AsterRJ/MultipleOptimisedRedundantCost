# MultipleOptimisedRedundantCost
Optimisation of multiple configuration space cost functions with Newton descent.

This is optimised with respect to a set list of cost functions, and a set robot.
The calculations were done with the package: https://github.com/philip-long/urdf_symbolic_models/tree/master
Which was modified in order to calculate derivatives of Jacobians from URDF files. Can be changed to listen to a parameter server, however the code from the linked repository needs more work to be done to handle differing joint types if this is the case.

Can be used to optimise path in OfflinePP, or in real time with OnlinePP - though this is not always globally optimal.


