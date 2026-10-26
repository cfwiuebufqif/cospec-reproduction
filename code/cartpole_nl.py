"""
Unified cart-pole: correct nonlinear dynamics with explicit inertia I,
numerically linearized at the upright equilibrium so the linear model used
for synthesis and the nonlinear plant used for simulation are identical.
Published parameters (Control Tutorials, CTM): M=0.5,m=0.5,b=0.1,l=0.3,I=0.006.
"""
import numpy as np, case_study as cs
G=9.81
CP=dict(M=0.5,m=0.5,b=0.1,l=0.3,I=0.006)
def _p(p):
    return p["I"]*(p["M"]+p["m"])+p["M"]*p["m"]*p["l"]*p["l"]
def deriv(state,F,p,mu_c=0.0,u_max=1e9,deadzone=0.0):
    x,xd,th,thd=state
    F=np.clip(F,-u_max,u_max)
    if abs(F)<deadzone: F=0.0
    Ft=F-p["b"]*xd-mu_c*np.sign(xd)
    s,c=np.sin(th),np.cos(th); P=_p(p)
    thdd=(p["m"]*G*p["l"]*(p["M"]+p["m"])*s - p["m"]*p["l"]*c*Ft
          - p["m"]*p["m"]*p["l"]*p["l"]*thd*thd*s*c)/P
    xdd=(Ft-p["m"]*p["l"]*thdd*c+p["m"]*p["l"]*thd*thd*s)/(p["M"]+p["m"])
    return np.array([xd,xdd,thd,thdd])
def linearize(p,dt=0.05):
    f0=deriv(np.zeros(4),0.0,p)
    Ac=np.zeros((4,4))
    eps=1e-6
    for j in range(4):
        e=np.zeros(4); e[j]=eps
        Ac[:,j]=(deriv(e,0.0,p)-f0)/eps
    Bc=((deriv(np.zeros(4),eps,p)-f0)/eps).reshape(4,1)
    return cs._zoh(Ac,Bc,dt)
def simulate(K,p,th0=0.15,steps=500,dt=0.05,mu_c=0.0,u_max=1e9,deadzone=0.0):
    s=np.array([0.0,0.0,th0,0.0])
    for _ in range(steps):
        F=-float((K@s).item())
        # RK4
        k1=deriv(s,F,p,mu_c,u_max,deadzone)
        k2=deriv(s+0.5*dt*k1,F,p,mu_c,u_max,deadzone)
        k3=deriv(s+0.5*dt*k2,F,p,mu_c,u_max,deadzone)
        k4=deriv(s+dt*k3,F,p,mu_c,u_max,deadzone)
        s=s+dt*(k1+2*k2+2*k3+k4)/6
        if abs(s[2])>1.5: return False,s[2]
    return abs(s[2])<0.05,s[2]
