import json,os,urllib.request,time
import numpy as np, case_study as cs
import anchoring_probe_v4 as ap
HOST="http://localhost:11434/v1/chat/completions"
MODEL=os.environ.get("SWEEP_MODEL","qwen2.5:7b")
def chat(prompt,temp):
    body=json.dumps({"model":MODEL,"temperature":temp,
        "messages":[{"role":"user","content":prompt}]}).encode()
    req=urllib.request.Request(HOST,data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]
def parse_cost(txt):
    import re
    # first try a JSON object anywhere in the response
    m=re.search(r"\{.*\}",txt,re.S)
    if m:
        try:
            d=json.loads(m.group(0))
            return float(d.get("q1")),float(d.get("q2")),float(d.get("r"))
        except Exception: pass
    def get(name):
        mm=re.search(name+r"\s*[=\:]\s*([0-9]*\.?[0-9]+)",txt)
        return float(mm.group(1)) if mm else None
    return get("q1"),get("q2"),get("r")
def admissible(A,Q,R):
    Q=(Q+Q.T)/2;R=(R+R.T)/2
    if np.min(np.linalg.eigvalsh(Q))<-1e-9:return False
    if np.min(np.linalg.eigvalsh(R))<=1e-9:return False
    n=A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam)>=1-1e-9:
            C=np.linalg.cholesky(Q+1e-12*np.eye(n)).T
            M=np.hstack([(A-lam*np.eye(n)).T,C.T])
            if np.linalg.matrix_rank(M,1e-7)<n:return False
    return True
WN,ZE=0.8,0.5
A,B=cs.second_order(WN,ZE)
V=ap.VIGNETTES["v1"]
from cost_probe import PROMPT
def run(vkeys=("v1","v6"),temps=(0.0,0.3,0.7,1.0),n=12):
    print("temperature | parse rate | cert reject | cost unstable | med beta/alpha")
    for temp in temps:
        tot=parse_fail=reject=unstable=0; ratios=[]
        for vk in vkeys:
            prompt=PROMPT.format(text=ap.VIGNETTES[vk]["text"])
            for _ in range(n):
                tot+=1
                try:
                    txt=chat(prompt,temp)
                    q1,q2,r=parse_cost(txt)
                except Exception:
                    parse_fail+=1; continue
                if q1 is None or q2 is None or r is None:
                    parse_fail+=1; continue
                Q=np.diag([q1,q2]); R=np.array([[r]])
                if not admissible(A,Q,R):
                    reject+=1; continue
                K=cs.lqr(A,B,Q,R)
                if cs.rho(A-B@K)>=1: unstable+=1
                ratios.append(max(q1,q2)/r)
        print("  %4.1f | %5.1f%% | %5.1f%% | %5.1f%% (n=%d) | %6.2f"%(
            temp,100*(tot-parse_fail)/tot,100*reject/tot,100*unstable/tot,tot,
            float(np.median(ratios)) if ratios else 0))
if __name__=="__main__":
    run()
