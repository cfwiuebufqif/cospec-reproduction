import numpy as np
import case_study as cs
from scipy.linalg import solve_discrete_lyapunov
WN,ZETA=0.8,0.5
A,B=cs.second_order(WN,ZETA); n=A.shape[0]
SQ=[0.2,1.0,5.0,50.0]; SR=[0.2,1.0,5.0,50.0]
def exact_omega_cross(sq,sr):
    def r(d):
        worst=0
        for sgn in (1,-1):
            Ah,Bh=cs.second_order(WN*(1+sgn*d),ZETA); K=cs.lqr(Ah,Bh,cs.Q_TASK*sq,cs.R_TASK*sr)
            worst=max(worst,cs.rho(A-B@K))
        return worst
    lo,hi=0,0.6
    for _ in range(44):
        m=(lo+hi)/2
        if r(m)<1: lo=m
        else: hi=m
    return lo
def sg_omega_guarantee(sq,sr):
    Ah0,Bh0=cs.second_order(WN,ZETA); K0=cs.lqr(Ah0,Bh0,cs.Q_TASK*sq,cs.R_TASK*sr)
    Phi=Ah0-Bh0@K0
    P=solve_discrete_lyapunov(Phi.T,np.eye(n))
    pn=np.linalg.norm(P,2); ph=np.linalg.norm(Phi,2)
    dmax=np.sqrt(ph*ph+1/pn)-ph
    def realized(d):
        worst=0
        for sgn in (1,-1):
            Ahm,Bhm=cs.second_order(WN*(1+sgn*d),ZETA); Km=cs.lqr(Ahm,Bhm,cs.Q_TASK*sq,cs.R_TASK*sr)
            worst=max(worst,np.linalg.norm((A-Ahm)-(B-Bhm)@Km,2))
        return worst
    lo,hi=0,0.6
    for _ in range(44):
        m=(lo+hi)/2
        if realized(m)<dmax: lo=m
        else: hi=m
    return lo
rows=[]
for sq in SQ:
    for sr in SR:
        ex=exact_omega_cross(sq,sr); sg=sg_omega_guarantee(sq,sr)
        rows.append((ex,sg,sq,sr))
rows.sort()
print("exact crossing / SG guarantee / sq / sr  (sorted by exact)")
for ex,sg,sq,sr in rows:
    print("  exact=%5.1f%%  SG-guaranteed=%5.1f%%  sq=%5s sr=%5s"%(100*ex,100*sg,sq,sr))
