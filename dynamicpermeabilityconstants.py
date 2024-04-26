''' 
Runscript for finding residues and poles based on the algorithm in Yvonne Ou, M.-J. (2014). On reconstruction of dynamic permeability and tortuosity from data at distinct frequencies. Inverse Problems, 30(9), 095002. https://doi.org/10.1088/0266-5611/30/9/095002

Setup with 4 sampling points

Parameter alt = {1: Cancellous bone,2: Cancellous bone, other: Sandstone} 


# Author: J.S.S
# Last revision: 26.04.24

'''



from scipy import signal
from scipy import optimize
import numpy as np
import cmath 


# Different parameter values
alt=1

if alt ==1:
	# Cancellous bone ex1 
	phi    = 0.67      # Porosity
	alpha  = 1.08      # Infitine frequency tortuisty
	K0     = 7*10**(-9)  # Static permeability
	eta    = 30*10**(-3) # Dynamic viscosity
	rho_f  = 1060      # Fluid density
	eta_k  = eta/rho_f # Kinectic viscosity
	Lambda = 10**(-5)  # Tuneable geometric constant
	F      = alpha/phi # Formation factor

elif alt ==2:
	# Cancellous bone ex2
	phi    = 0.8       # Porosity
	alpha  = 1.1       # Infitine frequency tortuisty
	K0     = 3e-8      # Static permeability
	eta    = 10**(-3)  # Dynamic viscosity
	rho_f  = 1000      # Fluid density
	eta_k  = eta/rho_f # Kinectic viscosity
	Lambda = 2.454e-5  # Tuneable geometric constant
	F      = alpha/phi # Formation factor
else:
	# Sandstone
	phi    = 0.2      # Porosity
	alpha  = 3.6      # Infitine frequency tortuisty
	K0     = 1e-13      # Static permeability
	eta    = 10**(-3)  # Dynamic viscosity
	rho_f  = 1040     # Fluid density
	eta_k  = eta/rho_f # Kinectic viscosity
	Lambda = 1.2e-7  # Tuneable geometric constant
	F      = alpha/phi # Formation factor


# Related constants
C2 = F*K0/eta_k
C1 = 4*C2*F*K0/Lambda**(2)




## JKD permeability in frequency domain
def JKD(w):
	
	A = 4*alpha**(2)*K0**(2)*rho_f*w
	B =eta*Lambda**(2)*phi**(2)
	C = A/B
	complex_number1 = complex(1,-C)
	D = alpha*K0*rho_f*w
	E =eta*phi
	G = D/E
	complex_number2 = complex(0,-G)
	a= F/eta_k
	val = a*K0/(cmath.sqrt(complex_number1)+complex_number2)**(1)
	
	return val
	
def PD(w):
	bop = C2*complex(0,-w)+cmath.sqrt(1+complex(0,-w)*C1) # w=is, then s=-iw
	pop = eta_k/F
	return C2/bop


M = 4                      # Number of sampling points
dw = 50/(M-1)              # Distance between spacing	
omega = np.arange(1,52,dw) # Sampling points in the range [1,51]Hz


# initialize arrays 
P = np.ones(4,dtype=complex)
P2 = np.ones(4,dtype=complex)

# Generate artificial data
for j in range(M):
	P[j]=JKD(omega[j])	# Data
	

P2 = np.conj(P)

# Create matrix	for 4 sampling points

a1 = np.array([1,complex(0,-omega[0]),complex(0,-omega[0])**(2),complex(0,-omega[0])**(3), 
	                -complex(0,-omega[0])*P[0],-complex(0,-omega[0])**(2)*P[0],-complex(0,-omega[0])**(3)*P[0],-complex(0,-omega[0])**(4)*P[0]])
a2 = np.array([1,complex(0,-omega[1]),complex(0,-omega[1])**(2),complex(0,-omega[1])**(3), 
	                -complex(0,-omega[1])*P[1],-complex(0,-omega[1])**(2)*P[1],-complex(0,-omega[1])**(3)*P[1],-complex(0,-omega[1])**(4)*P[1]])
a3 = np.array([1,complex(0,-omega[2]),complex(0,-omega[2])**(2),complex(0,-omega[2])**(3),
	                -complex(0,-omega[2])*P[2],-complex(0,-omega[2])**(2)*P[2],-complex(0,-omega[2])**(3)*P[2],-complex(0,-omega[2])**(4)*P[2]])
a4 = np.array([1,complex(0,-omega[3]),complex(0,-omega[3])**(2),complex(0,-omega[3])**(3),
	                -complex(0,-omega[3])*P[3],-complex(0,-omega[3])**(2)*P[3],-complex(0,-omega[3])**(3)*P[3],-complex(0,-omega[3])**(4)*P[3]])
# Conjugate pairs
a5 = np.array([1,np.conj(complex(0,-omega[0])),np.conj(complex(0,-omega[0]))**(2),np.conj(complex(0,-omega[0]))**(3),
	                -np.conj(complex(0,-omega[0]))**(1)*P2[0],-np.conj(complex(0,-omega[0]))**(2)*P2[0],-np.conj(complex(0,-omega[0]))**(3)*P2[0],-np.conj(complex(0,-omega[0]))**(4)*P2[0]])
a6 = np.array([1,np.conj(complex(0,-omega[1])),np.conj(complex(0,-omega[1]))**(2),np.conj(complex(0,-omega[1]))**(3),
	                -np.conj(complex(0,-omega[1]))**(1)*P2[1],-np.conj(complex(0,-omega[1]))**(2)*P2[1],-np.conj(complex(0,-omega[1]))**(3)*P2[1],-np.conj(complex(0,-omega[1]))**(4)*P2[1]])
a7 = np.array([1,np.conj(complex(0,-omega[2])),np.conj(complex(0,-omega[2]))**(2),np.conj(complex(0,-omega[2]))**(3),
	                -np.conj(complex(0,-omega[2]))**(1)*P2[2],-np.conj(complex(0,-omega[2]))**(2)*P2[2],-np.conj(complex(0,-omega[2]))**(3)*P2[2],-np.conj(complex(0,-omega[2]))**(4)*P2[2]])
a8 = np.array([1,np.conj(complex(0,-omega[3])),np.conj(complex(0,-omega[3]))**(2),np.conj(complex(0,-omega[3]))**(3),
	                -np.conj(complex(0,-omega[3]))**(1)*P2[3],-np.conj(complex(0,-omega[3]))**(2)*P2[3],-np.conj(complex(0,-omega[3]))**(3)*P2[3],-np.conj(complex(0,-omega[3]))**(4)*P2[3]])




A = np.array([a1,a2,a3,a4,a5,a6,a7,a8]) # Matrix
C = np.diag([1/np.linalg.norm(a1),1/np.linalg.norm(a2),1/np.linalg.norm(a3),1/np.linalg.norm(a4),1/np.linalg.norm(a5),1/np.linalg.norm(a6),1/np.linalg.norm(a7),1/np.linalg.norm(a8)])
B = np.matmul(A,C)

Bhat = np.concatenate([B.real,B.imag],axis = 0) # New matrix B hat



b = np.append(P,P2) # Vector
dhat = np.append(b.real,b.imag) # New vector d hat


guess = np.array([-1,1,0.7,0.5,1,1,-1,1]) # Initial guess

# Residual: Function being minimized wrt x
def residual(x):
	return np.matmul(Bhat,x)-dhat 
trt = optimize.least_squares(residual,guess,xtol=1e-15,gtol=1e-15,tr_solver='exact',method='trf') # Solve least square problem
vals = trt['x'] 

# Rescale
rescaled = np.matmul(C,vals) 

As = np.zeros(M,dtype=complex)
Bs = np.zeros(M,dtype=complex)

for k in range(len(As)):
	As[k] = rescaled[k]
	Bs[k] = rescaled[k+M]



coeff = np.array([Bs[3], Bs[2], Bs[1],Bs[0],1])
coeff2 = np.array([As[3], As[2], As[1],As[0]])

# Find residues and poles
r,p,k = signal.residue(coeff2,coeff) 
print('------------- Output ----------------')


real_residue = r.real
real_pole    = np.abs(p.real)
print(real_residue,'Residues')
print(real_pole,'Poles')
div = real_residue/real_pole
print('mu_0 =',sum(div))
print('mu_0 = 0.3986e-3 from Ou 2014')


