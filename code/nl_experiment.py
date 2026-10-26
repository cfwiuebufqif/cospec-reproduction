import numpy as np, case_study as cs, cartpole_nl as nl
Q=np.diag([1,1,10,10.0]); R=np.array([[0.1]])
def plant_gain(mult):
    wp=dict(nl.CP); wp["m"]=nl.CP["m"]*mult
    Aw,Bw=nl.linearize(wp); return cs.lqr(Aw,Bw,Q,R)
def cost_gain(mult):
    A,B=nl.linearize(nl.CP); s=abs(mult); return cs.lqr(A,B,Q*s,R*s)
def lin_rho(K):
    A,B=nl.linearize(nl.CP); return cs.rho(A-B@K)
print("model m multiplier | plant lin-rho | cost lin-rho")
for mult in [0.05,0.2,0.5,1.0]:
    Kp,Kc=plant_gain(mult),cost_gain(mult)
    print("  %5.2f | %6.3f | %6.3f"%(mult,lin_rho(Kp),lin_rho(Kc)))
print()
scenarios=[("no nonlinearity",dict(u_max=1e9,deadzone=0,mu_c=0)),
           ("saturation 15N",dict(u_max=15,deadzone=0,mu_c=0)),
           ("saturation 8N",dict(u_max=8,deadzone=0,mu_c=0)),
           ("sat 8N + deadzone 0.5 + Coulomb 0.1",dict(u_max=8,deadzone=0.5,mu_c=0.1))]
for label,sc in scenarios:
    print("--- %s ---"%label)
    for mult in [0.05,0.2,1.0]:
        Kp,Kc=plant_gain(mult),cost_gain(mult)
        op,fp=nl.simulate(Kp,nl.CP,th0=0.15,**sc)
        oc,fc=nl.simulate(Kc,nl.CP,th0=0.15,**sc)
        print("  m x%5.2f | plant: %-8s (%.2f) | cost: %-8s (%.3f)"%(
            mult,"UPRIGHT" if op else "FELL",fp,"UPRIGHT" if oc else "FELL",fc))
