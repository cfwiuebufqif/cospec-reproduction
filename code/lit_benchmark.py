"""
#37 literature-grounded benchmark plants.
Two canonical LQR systems with parameters taken from published tutorials:
  - cart-pole inverted pendulum, Control Tutorials (Univ. Alberta / UMich CTM):
      M=0.5, m=0.5, b=0.1, l=0.3, I=0.006, g=9.81
  - DC motor (CTM standard): J=0.01, b=0.1, K=0.01, R=1.0, L=0.5
For each, we perturb the parameters the LLM is asked for (same relative error
distribution measured in the logged study), route them two ways, and check the
true loop: plant routing (synthesise on the perturbed model, close the true
plant) vs cost routing (map the error onto weights, certify, synthesise on the
true plant). No new LLM call; the error distribution is the measured one.
"""
import numpy as np, case_study as cs
G=9.81
def cartpole(M,m,b,l,I,dt=0.05):
    p=I*(M+m)+M*m*l*l
    Ac=np.array([[0,1,0,0],
        [0,-(I+m*l*l)*b/p,(m*m*G*l*l)/p,0],
        [0,0,0,1],
        [0,-(m*l*b)/p,m*l*G*(M+m)/p,0]])
    Bc=np.array([[0],[(I+m*l*l)/p],[0],[m*l/p]])
    return cs._zoh(Ac,Bc,dt)
def dcmotor(J,b,K,R,L,dt=0.05):
    Ac=np.array([[0,1,0],[0,-b/J,K/J],[0,-K/L,-R/L]])
    Bc=np.array([[0],[0],[1/L]])
    return cs._zoh(Ac,Bc,dt)

def admissible(A,Qh,Rh):
    Qh=(Qh+Qh.T)/2; Rh=(Rh+Rh.T)/2
    if np.min(np.linalg.eigvalsh(Qh))<-1e-9: return False
    if np.min(np.linalg.eigvalsh(Rh))<=1e-9: return False
    n=A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam)>=1-1e-9:
            M=np.hstack([(A-lam*np.eye(n)).T, np.linalg.cholesky(Qh+1e-12*np.eye(n)).T])
            if np.linalg.matrix_rank(M,1e-7)<n: return False
    return True

def sweep(name, true_params, perturbed, make, Q, R, rel_errors, nrep=20):
    A,B=make(*true_params)
    plant_unstable=0; cost_unstable=0; cost_reject=0; tot=0
    rng=np.random.default_rng(7)
    for d in rel_errors:
        for _ in range(nrep):
            tot+=1
            # perturb only the listed parameter fields
            pt=list(true_params)
            for f in perturbed:
                sgn=rng.choice([-1,1]); pt[f]=true_params[f]*(1+sgn*d)
            Ah,Bh=make(*pt)
            # plant routing
            try:
                Kp=cs.lqr(Ah,Bh,Q,R)
                if cs.rho(A-B@Kp)>=1: plant_unstable+=1
            except Exception: plant_unstable+=1
            # cost routing: map relative error onto weights (monotone, same
            # convention as case study; scale Q by geometric mean of (1+err))
            scales=np.array([abs(pt[f]/true_params[f]) for f in perturbed])
            s=np.exp(np.mean(np.log(scales)))
            Qh=Q*s; Rh=R*s
            if not admissible(A,Qh,Rh):
                cost_reject+=1; continue
            Kc=cs.lqr(A,B,Qh,Rh)
            if cs.rho(A-B@Kc)>=1: cost_unstable+=1
    print("%-10s plant routing unstable=%5.1f%% | cost routing unstable=%5.1f%% (reject %5.1f%%)"%(
        name,100*plant_unstable/tot,100*cost_unstable/tot,100*cost_reject/tot))

# cart-pole: LLM is asked for pendulum length l (index3) and mass m (index1)
from collections import namedtuple
CP=namedtuple("CP",["M","m","b","l","I"])(0.5,0.5,0.1,0.3,0.006)
DM=namedtuple("DM",["J","b","K","R","L"])(0.01,0.1,0.01,1.0,0.5)
rel=np.linspace(0.05,0.6,8)
Qcp=np.diag([1,1,10,10.0]); Rcp=np.array([[0.1]])
Qdm=np.diag([10,1,1.0]); Rdm=np.array([[0.1]])
if __name__=="__main__":
    # Mild, linear-only relative-error sweep (5%--60%, the bulk of the logged
    # distribution). Both routings stay stable here; instability appears only at
    # the order-of-magnitude errors and in the nonlinear plant. Those stronger
    # tests are lit_cross.py (linear extremes) and scan_all_params.py (Table 3).
    sweep("cart-pole",CP,perturbed=[1,3],make=cartpole,Q=Qcp,R=Rcp,rel_errors=rel)
    sweep("DC-motor",DM,perturbed=[0,2],make=dcmotor,Q=Qdm,R=Rdm,rel_errors=rel)
