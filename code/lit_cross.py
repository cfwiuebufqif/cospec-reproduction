import numpy as np, case_study as cs
from lit_benchmark import cartpole,dcmotor,CP,DM
def plant_rho(make,true,field,mult,Q,R):
    A,B=make(*true)
    pt=list(true); pt[field]=true[field]*mult
    Ah,Bh=make(*pt)
    K=cs.lqr(Ah,Bh,Q,R)
    return cs.rho(A-B@K)
def cost_rho(make,true,field,mult,Q,R):
    A,B=make(*true)
    s=abs(mult)
    K=cs.lqr(A,B,Q*s,R*s)
    return cs.rho(A-B@K)
Qcp=np.diag([1,1,10,10.0]); Rcp=np.array([[0.1]])
Qdm=np.diag([10,1,1.0]); Rdm=np.array([[0.1]])
print("cart-pole: perturb pendulum mass m (field1), true=0.5")
print(" mult (model m / true m) | plant-routing rho | cost-routing rho")
for mult in [0.05,0.1,0.2,0.4,0.7,1.0,1.5,3,10]:
    pr=plant_rho(cartpole,CP,1,mult,Qcp,Rcp)
    cr=cost_rho(cartpole,CP,1,mult,Qcp,Rcp)
    print("  %6.2f | %6.3f %s | %6.3f"%(mult,pr,"UNSTABLE" if pr>=1 else "        ",cr))
print("\nDC motor: perturb torque constant K (field2), true=0.01")
for mult in [0.05,0.1,0.2,0.5,1.0,2,5,10]:
    pr=plant_rho(dcmotor,DM,2,mult,Qdm,Rdm)
    cr=cost_rho(dcmotor,DM,2,mult,Qdm,Rdm)
    print("  %6.2f | %6.3f %s | %6.3f"%(mult,pr,"UNSTABLE" if pr>=1 else "        ",cr))
