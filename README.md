# Dynamic Permeability Constants

This repository contains the implementation related to the article

Stokke, J.S., Bause, M., Margenberg, N., Radu, F.A., Biot’s poro-elasticity system with dynamic permeability convolution: Well-posedness for evolutionary form

https://www.sciencedirect.com/science/article/pii/S0893965924002441 


---

Replication of the algorithm in Yvonne Ou, M.-J. (2014). On reconstruction of dynamic permeability and tortuosity from data at distinct frequencies. Inverse Problems, 30(9), 095002. https://doi.org/10.1088/0266-5611/30/9/095002. The least-square problem is solved using scipy.optimize.least_squares.

The runscript includes different material parameters, two for cancellous bone and one for sandstone. It outputs the constants c_j (poles) and d_j (residues).
