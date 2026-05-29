import json, sys, time, pickle, logging, os
import numpy as np
from aco import ACOConfig, TSPProblem, DistributedMMAS, tour_length
logging.disable(logging.CRITICAL)
AREA=1000.0*1000.0; BHH=0.7124
bhh=lambda n: BHH*np.sqrt(n*AREA)
def make(n,s):
    rng=np.random.default_rng(s); return TSPProblem(rng.uniform(0,1000,(n,2)),nn_size=16)
def run(p,**kw):
    cfg=ACOConfig(**kw); t=time.perf_counter()
    with DistributedMMAS(p,cfg) as e: r=e.solve()
    return r, time.perf_counter()-t
RES="results.json"
def load(): return json.load(open(RES)) if os.path.exists(RES) else {}
def save(d): json.dump(d,open(RES,"w"),indent=2)
SEEDS=[0,1,2]

def e1(n):
    d=load(); d.setdefault("quality",[])
    nn,best,gaps,times=[],[],[],[]
    for s in SEEDS:
        p=make(n,s); nn.append(tour_length(p.nearest_neighbour_tour(),p.dist))
        r,dt=run(p,n_ants=40,n_workers=4,max_iterations=100,use_2opt=True,seed=s)
        best.append(r.best_length); gaps.append(100*(r.best_length-bhh(n))/bhh(n)); times.append(dt)
    d["quality"].append(dict(n=n,bhh=bhh(n),nn_mean=float(np.mean(nn)),
        best_mean=float(np.mean(best)),best_std=float(np.std(best)),
        gap_bhh_mean=float(np.mean(gaps)),
        improve_vs_nn=float(100*(np.mean(nn)-np.mean(best))/np.mean(nn)),
        time_mean=float(np.mean(times))))
    save(d); print(f"n={n}: best={np.mean(best):.1f}+/-{np.std(best):.1f}")

def e2():
    d=load(); e={}
    for use in (False,True):
        best,times=[],[]
        for s in SEEDS:
            p=make(200,s); r,dt=run(p,n_ants=40,n_workers=4,max_iterations=100,use_2opt=use,seed=s)
            best.append(r.best_length); times.append(dt)
        e["with_2opt" if use else "no_2opt"]=dict(best_mean=float(np.mean(best)),
            best_std=float(np.std(best)),gap_bhh=float(100*(np.mean(best)-bhh(200))/bhh(200)),
            time_mean=float(np.mean(times)))
    d["ablation_2opt"]=e; save(d)

def e3():
    d=load(); arr=[]
    for na in [16,32,64]:
        best=[]
        for s in SEEDS:
            p=make(200,s); r,_=run(p,n_ants=na,n_workers=4,max_iterations=100,use_2opt=True,seed=s)
            best.append(r.best_length)
        arr.append(dict(n_ants=na,best_mean=float(np.mean(best)),best_std=float(np.std(best))))
    d["ant_sensitivity"]=arr; save(d)

def e4():
    d=load(); conv={}
    for use in (True,False):
        p=make(200,0); cfg=ACOConfig(n_ants=40,n_workers=4,max_iterations=120,use_2opt=use,seed=0)
        with DistributedMMAS(p,cfg) as e: r=e.solve()
        conv["with_2opt" if use else "no_2opt"]=r.history
    d["convergence"]=conv; save(d)

def e5():
    d=load(); p=make(300,0); pher=p.dist.nbytes
    tb=len(pickle.dumps((10,12345),protocol=pickle.HIGHEST_PROTOCOL)); nw=4
    d["communication"]=dict(n=300,n_workers=nw,pheromone_bytes=int(pher),
        our_payload_per_iter=int(tb*nw),naive_payload_per_iter=int(pher*nw),
        reduction_factor=float(pher/tb)); save(d)

globals()[sys.argv[1]](*[int(x) for x in sys.argv[2:]])
