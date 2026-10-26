import numpy as np, case_study as cs, cartpole_nl as nl
Q=np.diag([1,1,10,10.0]); R=np.array([[0.1]])
fields=["M","m","l","I"]
def plant_gain(field,mult):
    wp=dict(nl.CP); wp[field]=nl.CP[field]*mult
    Aw,Bw=nl.linearize(wp); return cs.lqr(Aw,Bw,Q,R)
def cost_gain(field,mult):
    A,B=nl.linearize(nl.CP); s=abs(mult); return cs.lqr(A,B,Q*s,R*s)
A0,B0=nl.linearize(nl.CP)
print("cases where PLANT routing fails; does saturation help?")
print(" field | mult | lin-rho | no-sat | sat15 | sat8 | cost no-sat | cost sat8")
for field in fields:
    for mult in [0.02,0.05,0.1,0.2,0.5,2,5,10,20]:
        Kp=plant_gain(field,mult); Kc=cost_gain(field,mult)
        lr=cs.rho(A0-B0@Kp)
        on,_=nl.simulate(Kp,nl.CP,th0=0.15,u_max=1e9)
        o15,_=nl.simulate(Kp,nl.CP,th0=0.15,u_max=15)
        o8,_=nl.simulate(Kp,nl.CP,th0=0.15,u_max=8)
        cn,_=nl.simulate(Kc,nl.CP,th0=0.15,u_max=1e9)
        c8,_=nl.simulate(Kc,nl.CP,th0=0.15,u_max=8)
        flag = (lr>=1) or (not on) or (not o15) or (not o8)
        if flag:
            print("  %s | %6.2f | %6.3f | %-5s | %-5s | %-5s | %-5s | %-5s"%(
                field,mult,lr,
                "UP" if on else "FELL","UP" if o15 else "FELL","UP" if o8 else "FELL",
                "UP" if cn else "FELL","UP" if c8 else "FELL"))
