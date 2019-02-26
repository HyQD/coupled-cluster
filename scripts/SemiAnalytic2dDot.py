import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

def ddx_psi(y,dx):
    N = len(y)
    ddx_y = np.zeros(N,dtype=np.complex128)
    ddx_y[1:N-1] = (y[0:N-2]-2*y[1:N-1]+y[2:])/dx**2
    return ddx_y

def f(t,y,X,Omega,w,E0=0.5):
    dx = X[1]-X[0]
    rhs = -0.5*ddx_psi(y,dx)+0.5*Omega**2*X**2*y+4*E0*X*np.sin(w*t)*y
    return 0.5*rhs

Omega = 1
wR = 2*Omega
wr = 0.5*Omega

n = 2
m = 0

eps_r = (abs(m)+2)*wr

print("eps_r",eps_r)
wat

eta_X = 0.5*wR
eta_Y = 0.5*wR
eta_XY = eta_X+eta_Y

print(eta_XY,eps_r)
print(2*eps_r+0.5*eta_X+0.5*eta_Y)


r = np.linspace(0,5,201)
u_r = 1.0/np.sqrt(2*np.pi)*np.exp(-0.25*r**2)*(1+r)
u_r /= np.sqrt(trapz(np.abs(u_r)**2,r))

#Setup center of mass intitial condition
N    = 400
rmax = 14
R = np.linspace(-rmax,rmax,N)
psiX = (wR/np.pi)**0.25 * np.exp(-0.5*wR*R**2)

# Time parameters
Psi = psiX.copy()
T = 10
dt = 1e-3 
counter = 0
time_steps = int( T / dt ) # Number of time steps
overlap = [simps(np.conj(Psi)*Psi,R)]
t_list = [0]
Energy = []
w = 8*Omega

Energy.append(0.5*eta_X)
Psi = np.complex128(Psi)

while counter < time_steps:
    
    counter += 1
    tn = counter*dt
    k1 = -dt*1j*f(tn,Psi,R,wR,w)
    k2 = -dt*1j*f(tn+0.5*dt,Psi+0.5*k1,R,wR,w)
    k3 = -dt*1j*f(tn+0.5*dt,Psi+0.5*k2,R,wR,w)
    k4 = -dt*1j*f(tn+dt,Psi+k3,R,wR,w)

    Psi = Psi + 1.0/6.0*(k1+2*(k2+k3)+k4)
    Ht_psi = f(tn,Psi,R,wR,w)
    Energy.append(simps(np.conj(Psi)*Ht_psi,R).real)
    
    overlap.append(np.abs(simps(np.conj(Psi)*psiX,R))**2)
    t_list.append(tn)
    #print(trapz(Psi.conj()*Psi,R),tn)

plt.figure(1)
plt.plot(t_list,np.array(Energy)+0.5*eta_Y+2*eps_r)

plt.figure(2)
plt.plot(t_list,overlap)
plt.show()



