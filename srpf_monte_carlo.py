#!/usr/bin/env python3
"""
SRPF Monte Carlo Validation — v3 (physics-correct discrimination)
X. J. Régent, 2026

Key insight: the spectral separation manifests in the DETRENDED
innovation residuals. Raw innovations grow over time for all targets;
after removing the polynomial trend (which captures the systematic
divergence), the residuals reveal the true structural difference:
  - Ballistic: small, white-noise residuals (good model fit)
  - Non-ballistic: large, correlated residuals (poor model fit)
"""
import numpy as np
from numpy.fft import rfft
from scipy import stats
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#FAFAFA",
    "axes.grid":True,"grid.alpha":0.3,"font.size":10,
    "axes.titlesize":11,"legend.fontsize":8,"figure.dpi":150,
})
CB="#1B6CA8"; CC="#D94032"; CD="#E8872B"; CS="#2A9D5C"; CG="#999"; CA="#6B3FA0"

# ═══════════ PHYSICS ═══════════

def rk4_bal(s, dt, g=9.81, rho0=1.225, Hs=8500, beta=5556):
    def f(s):
        x,y,vx,vy = s
        sp = np.sqrt(vx**2+vy**2)+1e-10
        rho = rho0*np.exp(-max(y,0)/Hs)
        d = rho/(2*beta)
        return np.array([vx, vy, -d*sp*vx, -g - d*sp*vy])
    k1=f(s); k2=f(s+.5*dt*k1); k3=f(s+.5*dt*k2); k4=f(s+dt*k3)
    return s + dt/6*(k1+2*k2+2*k3+k4)

def gen_bal(N, dt, v0=300, ang=55, Qstd=0.3):
    a = np.radians(ang)
    s = np.array([0., 100., v0*np.cos(a), v0*np.sin(a)])
    pos = np.zeros((N,2))
    for k in range(N):
        pos[k] = s[:2]; s = rk4_bal(s, dt)
        s[2:] += np.random.randn(2)*Qstd
    return pos

def gen_clutter(N, dt, drift=30, wander=120):
    pos = np.zeros((N,2))
    d = np.random.randn(2); d /= (np.linalg.norm(d)+1e-10)
    pos[0] = [np.random.uniform(500,4000), np.random.uniform(500,3000)]
    for k in range(1,N):
        pos[k] = pos[k-1] + d*drift*dt + np.random.randn(2)*wander*np.sqrt(dt)
    return pos

def gen_decoy(N, dt, v0=260, ang=50, lift=50):
    a = np.radians(ang)
    s = np.array([0., 100., v0*np.cos(a), v0*np.sin(a)])
    pos = np.zeros((N,2))
    for k in range(N):
        pos[k] = s[:2]; s = rk4_bal(s, dt)
        s[2] += lift*np.sin(.5*k*dt)*dt + lift*.5*np.random.randn()*dt
        s[3] += lift*np.cos(.3*k*dt)*dt + lift*.5*np.random.randn()*dt
    return pos

def add_noise(pos, R=15.0):
    return pos + np.random.randn(*pos.shape)*R

# ═══════════ INNOVATION & SCORES ═══════════

def innov_2d(meas, dt):
    """Return 2D innovation vectors (not just norms)."""
    N = len(meas)
    v0 = (meas[1]-meas[0])/dt
    s = np.array([meas[0,0], meas[0,1], v0[0], v0[1]])
    innov = np.zeros((N,2))
    for k in range(N):
        innov[k] = meas[k] - s[:2]
        s = rk4_bal(s, dt)
    return innov

def detrend(seq, order=2):
    """Remove polynomial trend of given order."""
    N = len(seq)
    t = np.arange(N, dtype=float)
    coeffs = np.polyfit(t, seq, order)
    trend = np.polyval(coeffs, t)
    return seq - trend

def srpf_score(meas, dt):
    """
    SRPF Ballistic Resonance Score (Definition 4.1).
    
    Multi-feature classifier combining:
    1. Normalised innovation energy (low for ballistic)
    2. Whiteness test on detrended x-residuals  
    3. Whiteness test on detrended y-residuals
    4. Spectral flatness of detrended residuals
    
    Returns score in [0,1], higher = more ballistic.
    """
    innov = innov_2d(meas, dt)
    N = len(innov)
    
    # 1. Normalised energy: E = mean(||δ_k||) / R_expected
    R_expected = 30.0  # ~2*R_std for 2D norm of N(0,R²)
    norms = np.linalg.norm(innov, axis=1)
    f_energy = np.exp(-np.mean(norms) / R_expected)
    
    # Detrend both components (remove systematic divergence)
    dx = detrend(innov[:,0], order=2)
    dy = detrend(innov[:,1], order=2)
    
    # 2-3. Ljung-Box-style whiteness: autocorrelation at lags 1-5
    def whiteness(r):
        if np.std(r) < 1e-10: return 0.5
        r_n = (r - np.mean(r)) / (np.std(r) + 1e-10)
        ac = np.array([np.corrcoef(r_n[:-lag], r_n[lag:])[0,1]
                       for lag in range(1, min(6, N-1))])
        # For white noise, |autocorrelation| should be small
        # Q ≈ N * Σ ρ²(lag) ~ χ²(5) under H0
        Q = N * np.sum(ac**2)
        # Transform: low Q (white) → high score
        return np.exp(-Q / (2*N))
    
    f_white_x = whiteness(dx)
    f_white_y = whiteness(dy)
    
    # 4. Spectral flatness of detrended residuals
    # Wiener entropy: geometric mean / arithmetic mean of PSD
    # Flat spectrum → ratio ≈ 1 (white noise); peaked → ratio << 1
    def spectral_flatness(r):
        spec = np.abs(rfft(r))**2
        spec = spec[1:]  # exclude DC
        if len(spec) < 2 or np.mean(spec) < 1e-30:
            return 0.5
        log_mean = np.mean(np.log(spec + 1e-30))
        return np.exp(log_mean) / (np.mean(spec) + 1e-30)
    
    f_sf_x = spectral_flatness(dx)
    f_sf_y = spectral_flatness(dy)
    f_sf = 0.5*(f_sf_x + f_sf_y)
    
    # Combine: all features ∈ [0,1], higher = more ballistic
    return 0.35*f_energy + 0.25*f_white_x + 0.25*f_white_y + 0.15*f_sf

# ═══════════ EXPERIMENTS ═══════════

def exp1(n_trials=500, N=128, dt=0.05):
    print("="*65)
    print("  EXP 1: Spectral Separation on DETRENDED Residuals")
    print("="*65)
    half = N//2+1
    psd_b=np.zeros(half); psd_c=np.zeros(half); psd_d=np.zeros(half)
    cb=cc=cd=0
    for t in range(n_trials):
        if (t+1)%100==0: print(f"  {t+1}/{n_trials}...")
        # Ballistic
        try:
            iv=innov_2d(add_noise(gen_bal(N,dt,v0=np.random.uniform(200,400),
                        ang=np.random.uniform(35,70))),dt)
            dx=detrend(iv[:,0]); dy=detrend(iv[:,1])
            r=np.sqrt(dx**2+dy**2)
            psd_b+=np.abs(rfft(r-np.mean(r)))**2/N; cb+=1
        except: pass
        # Clutter
        try:
            iv=innov_2d(add_noise(gen_clutter(N,dt)),dt)
            dx=detrend(iv[:,0]); dy=detrend(iv[:,1])
            r=np.sqrt(dx**2+dy**2)
            psd_c+=np.abs(rfft(r-np.mean(r)))**2/N; cc+=1
        except: pass
        # Decoy
        try:
            iv=innov_2d(add_noise(gen_decoy(N,dt,v0=np.random.uniform(220,350),
                        ang=np.random.uniform(40,65))),dt)
            dx=detrend(iv[:,0]); dy=detrend(iv[:,1])
            r=np.sqrt(dx**2+dy**2)
            psd_d+=np.abs(rfft(r-np.mean(r)))**2/N; cd+=1
        except: pass
    
    psd_b/=max(cb,1); psd_c/=max(cc,1); psd_d/=max(cd,1)
    j=np.arange(2,min(30,half))
    sb=stats.linregress(np.log10(j),np.log10(psd_b[j]+1e-30)).slope
    sc=stats.linregress(np.log10(j),np.log10(psd_c[j]+1e-30)).slope
    sd=stats.linregress(np.log10(j),np.log10(psd_d[j]+1e-30)).slope
    print(f"\n  Detrended PSD slopes: bal={sb:.3f}, cl={sc:.3f}, dec={sd:.3f}")
    print(f"  PSD levels (j=2-10): bal={np.mean(psd_b[2:10]):.0f}, "
          f"cl={np.mean(psd_c[2:10]):.0f}, dec={np.mean(psd_d[2:10]):.0f}")
    print(f"  Level ratio cl/bal: {np.mean(psd_c[2:10])/np.mean(psd_b[2:10]+1e-30):.1f}x")
    return {"psd_b":psd_b,"psd_c":psd_c,"psd_d":psd_d,"slopes":(sb,sc,sd),"N":half}


def exp2(n_trials=2000, N=128, dt=0.05):
    print("\n"+"="*65)
    print("  EXP 2: SRPF Score Distributions & ROC")
    print("="*65)
    sb_s,sc_s,sd_s=[],[],[]
    for t in range(n_trials):
        if (t+1)%500==0: print(f"  {t+1}/{n_trials}...")
        try:
            m=add_noise(gen_bal(N,dt,v0=np.random.uniform(200,400),
                        ang=np.random.uniform(35,70)))
            sb_s.append(srpf_score(m,dt))
        except: pass
        try:
            m=add_noise(gen_clutter(N,dt))
            sc_s.append(srpf_score(m,dt))
        except: pass
        try:
            m=add_noise(gen_decoy(N,dt,lift=50))
            sd_s.append(srpf_score(m,dt))
        except: pass

    sb_s=np.array(sb_s); snb=np.concatenate([sc_s,sd_s])
    mu1,s1=np.mean(sb_s),np.std(sb_s)
    mu0,s0=np.mean(snb),np.std(snb)
    d2=(mu1-mu0)**2/(s0**2+1e-20)
    print(f"\n  Ballistic:     μ={mu1:.4f}, σ={s1:.4f}")
    print(f"  Non-ballistic: μ={mu0:.4f}, σ={s0:.4f}")
    print(f"  Δμ={mu1-mu0:.4f}, d²={d2:.2f}")

    lo=min(snb.min(),sb_s.min()); hi=max(snb.max(),sb_s.max())
    thr=np.linspace(lo,hi,500)
    PD=np.array([np.mean(sb_s>=t) for t in thr])
    PFA=np.array([np.mean(snb>=t) for t in thr])
    idx=np.argsort(PFA); auc=trapezoid(PD[idx],PFA[idx])

    # Energy baseline
    np.random.seed(777)
    eb,enb=[],[]
    for _ in range(min(n_trials,1000)):
        try:
            iv=innov_2d(add_noise(gen_bal(N,dt,v0=np.random.uniform(200,400),
                        ang=np.random.uniform(35,70))),dt)
            eb.append(np.mean(np.linalg.norm(iv,axis=1)))
        except: pass
        try:
            pos=gen_clutter(N,dt) if np.random.rand()<.5 else gen_decoy(N,dt,lift=50)
            iv=innov_2d(add_noise(pos),dt)
            enb.append(np.mean(np.linalg.norm(iv,axis=1)))
        except: pass
    eb=np.array(eb); enb=np.array(enb)
    thr_e=np.linspace(min(enb.min(),eb.min()),max(enb.max(),eb.max()),500)
    PD_e=np.array([np.mean(eb<=t) for t in thr_e])
    PFA_e=np.array([np.mean(enb<=t) for t in thr_e])
    idx_e=np.argsort(PFA_e); auc_e=trapezoid(PD_e[idx_e],PFA_e[idx_e])

    print(f"  AUC SRPF:   {auc:.4f}")
    print(f"  AUC Energy: {auc_e:.4f}")
    print(f"  Winner:     {'SRPF ✓' if auc>auc_e else 'Energy'}")
    return {"sb":sb_s,"sc":np.array(sc_s),"sd":np.array(sd_s),"snb":snb,
            "thr":thr,"PD":PD,"PFA":PFA,"PD_e":PD_e,"PFA_e":PFA_e,
            "auc":auc,"auc_e":auc_e,"stats":(mu1,s1,mu0,s0)}


def exp3(n_trials=1500, N=128, dt=0.05, alpha=0.3):
    print("\n"+"="*65)
    print(f"  EXP 3: Gain G(τ) [α={alpha}]")
    print("="*65)
    data=[]
    for t in range(n_trials):
        if (t+1)%300==0: print(f"  {t+1}/{n_trials}...")
        is_b=np.random.rand()<alpha
        try:
            if is_b:
                pos=gen_bal(N,dt,v0=np.random.uniform(200,400),ang=np.random.uniform(35,70))
            else:
                pos=gen_clutter(N,dt) if np.random.rand()<.5 else gen_decoy(N,dt,lift=50)
            m=add_noise(pos)
            data.append((srpf_score(m,dt),is_b))
        except: pass

    scores=np.array([x[0] for x in data]); labels=np.array([x[1] for x in data])
    taus=np.linspace(np.percentile(scores,2),np.percentile(scores,98),80)
    res=[]
    for tau in taus:
        p=scores>=tau; nb=np.sum(labels)
        f=np.mean(p); g=np.sum(p&labels)/nb if nb else 0
        G=g/f if f>1e-10 else 0
        pfa=np.sum(p&~labels)/np.sum(~labels) if np.sum(~labels) else 0
        res.append({"tau":tau,"f":f,"g":g,"G":G,"filt":1-f,"pfa":pfa})

    valid=[r for r in res if r["g"]>=0.90]
    best=max(valid,key=lambda r:r["G"]) if valid else max(res,key=lambda r:r["G"])
    print(f"\n  τ*={best['tau']:.4f}, filter={best['filt']*100:.1f}%, "
          f"G=×{best['G']:.2f}, P_D={best['g']*100:.1f}%, P_FA={best['pfa']*100:.1f}%")
    return {"res":res,"best":best,"scores":scores,"labels":labels}


def exp4(n_scen=500, M=500, alpha=0.1, C=100, N=64, dt=0.05):
    print("\n"+"="*65)
    print(f"  EXP 4: Saturation (M={M}, α={alpha}, C={C})")
    print("="*65)
    sb,snb=[],[]
    for _ in range(400):
        try:
            m=add_noise(gen_bal(N,dt,v0=np.random.uniform(200,400),ang=np.random.uniform(35,70)))
            sb.append(srpf_score(m,dt))
        except: pass
        try:
            pos=gen_clutter(N,dt) if np.random.rand()<.5 else gen_decoy(N,dt,lift=50)
            m=add_noise(pos)
            snb.append(srpf_score(m,dt))
        except: pass
    sb=np.array(sb); snb=np.array(snb)
    tau=np.percentile(sb,5)
    PD=np.mean(sb>=tau); PFA=np.mean(snb>=tau)
    print(f"  τ={tau:.4f}, P_D={PD:.3f}, P_FA={PFA:.3f}")

    Mb=int(M*alpha); Mnb=M-Mb
    mn,ms,ml=[],[],[]
    for _ in range(n_scen):
        if M>C: det_no=np.random.binomial(Mb,C/M)
        else: det_no=Mb
        mn.append(Mb-det_no)
        bp=np.random.binomial(Mb,PD); np_=np.random.binomial(Mnb,PFA)
        tot=bp+np_; ml.append(tot)
        if tot<=C: det=bp
        else: det=np.random.binomial(bp,C/tot)
        ms.append(Mb-det)

    print(f"  WITHOUT: missed={np.mean(mn):.1f}/{Mb}")
    print(f"  WITH:    missed={np.mean(ms):.1f}/{Mb}, load={np.mean(ml):.1f}/{C}")
    return {"mn":mn,"ms":ms,"ml":ml,
            "p":{"M":M,"alpha":alpha,"C":C,"tau":tau,"PD":PD,"PFA":PFA,"Mb":Mb}}


# ═══════════ PLOT ═══════════

def plot_all(e1,e2,e3,e4,path):
    fig=plt.figure(figsize=(18,22))
    gs=GridSpec(4,2,hspace=.38,wspace=.30,left=.07,right=.96,top=.94,bottom=.04)

    ax=fig.add_subplot(gs[0,0])
    j=np.arange(2,min(40,e1["N"]))
    ax.loglog(j,e1["psd_b"][j],color=CB,lw=2.2,label="Ballistic")
    ax.loglog(j,e1["psd_c"][j],color=CC,lw=2.2,label="Clutter")
    ax.loglog(j,e1["psd_d"][j],color=CD,lw=2.2,label="Decoy")
    ax.set_xlabel("Frequency $j$"); ax.set_ylabel("PSD (detrended)")
    ax.set_title("(a) Detrended Innovation PSD"); ax.legend()
    sl=e1["slopes"]
    ax.text(.03,.06,f"slopes: b={sl[0]:.2f} c={sl[1]:.2f} d={sl[2]:.2f}",
            transform=ax.transAxes,fontsize=8,bbox=dict(boxstyle="round",fc="w",alpha=.8))

    ax=fig.add_subplot(gs[0,1])
    lo=min(e2["snb"].min(),e2["sb"].min()); hi=max(e2["snb"].max(),e2["sb"].max())
    bins=np.linspace(lo,hi,60)
    ax.hist(e2["sb"],bins,alpha=.6,color=CB,density=True,
            label=f'Ballistic ($\\mu$={e2["stats"][0]:.3f})')
    ax.hist(e2["snb"],bins,alpha=.6,color=CC,density=True,
            label=f'Non-bal ($\\mu$={e2["stats"][2]:.3f})')
    ax.axvline(e3["best"]["tau"],color="k",ls="--",lw=1.5,
               label=f'$\\tau^*$={e3["best"]["tau"]:.3f}')
    ax.set_xlabel("SRPF Score"); ax.set_ylabel("Density")
    ax.set_title("(b) Score Distributions"); ax.legend()

    ax=fig.add_subplot(gs[1,0])
    ax.plot(e2["PFA"],e2["PD"],color=CS,lw=2.5,label=f'SRPF (AUC={e2["auc"]:.3f})')
    ax.plot(e2["PFA_e"],e2["PD_e"],color=CG,lw=2,ls="--",label=f'Energy (AUC={e2["auc_e"]:.3f})')
    ax.plot([0,1],[0,1],"k:",lw=.8,alpha=.3)
    ax.set_xlabel("$P_{FA}$"); ax.set_ylabel("$P_D$")
    ax.set_title("(c) ROC: SRPF vs Energy Detection"); ax.legend(loc="lower right")
    ax.set_xlim(-.02,1.02); ax.set_ylim(-.02,1.02)

    ax=fig.add_subplot(gs[1,1])
    taus=[r["tau"] for r in e3["res"]]; Gs=[r["G"] for r in e3["res"]]
    filt=[r["filt"]*100 for r in e3["res"]]; pds=[r["g"]*100 for r in e3["res"]]
    ax2=ax.twinx()
    l1,=ax.plot(taus,Gs,color=CS,lw=2.5,label="Gain $\\mathcal{G}$")
    l2,=ax2.plot(taus,filt,color=CA,lw=2,ls="--",label="Filter %")
    l3,=ax2.plot(taus,pds,color=CB,lw=2,ls=":",label="$P_D$ %")
    ax.axhline(1,color="gray",ls=":",lw=.8)
    ax.set_xlabel("$\\tau$"); ax.set_ylabel("Gain",color=CS); ax2.set_ylabel("%",color=CA)
    ax.set_title("(d) Gain $\\mathcal{G}(\\tau)$")
    ax.legend([l1,l2,l3],[l.get_label() for l in [l1,l2,l3]],loc="center left",fontsize=8)

    ax=fig.add_subplot(gs[2,0])
    pp=e4["p"]; mn_=np.mean(e4["mn"]); ms_=np.mean(e4["ms"]); ml_=np.mean(e4["ml"])
    x=np.arange(2); w=.3
    b1=ax.bar(x-w/2,[mn_,pp["C"]],w,color=CC,alpha=.8,label="Without SRPF")
    b2=ax.bar(x+w/2,[ms_,ml_],w,color=CS,alpha=.8,label="With SRPF")
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.8,f"{b.get_height():.1f}",ha="center",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["Threats\nMissed","EKF\nLoad"])
    ax.axhline(pp["C"],color="red",ls=":",lw=1,alpha=.5)
    ax.set_title(f"(e) Saturation: M={pp['M']}, threats={pp['Mb']}, C={pp['C']}"); ax.legend()

    ax=fig.add_subplot(gs[2,1])
    mx=max(max(e4["mn"]),max(e4["ms"]))+1; bins=np.arange(-.5,mx+1,1)
    ax.hist(e4["mn"],bins,alpha=.6,color=CC,density=True,label=f'Without ($\\mu$={np.mean(e4["mn"]):.1f})')
    ax.hist(e4["ms"],bins,alpha=.6,color=CS,density=True,label=f'With ($\\mu$={np.mean(e4["ms"]):.1f})')
    ax.set_xlabel("Threats Missed"); ax.set_ylabel("Density"); ax.set_title("(f) Missed Threats"); ax.legend()

    ax=fig.add_subplot(gs[3,0])
    np.random.seed(42)
    for i in range(3):
        pos=gen_bal(250,.05,v0=250+i*50,ang=45+i*8); m=pos[:,1]>=0
        ax.plot(pos[m,0]/1e3,pos[m,1]/1e3,color=CB,lw=1.5,alpha=.5+.2*i,label="Ballistic" if i==0 else None)
    for i in range(3):
        pos=gen_clutter(250,.05)
        ax.plot(pos[:,0]/1e3,pos[:,1]/1e3,color=CC,lw=1,alpha=.5,label="Clutter" if i==0 else None)
    for i in range(2):
        pos=gen_decoy(250,.05,v0=240+i*30,lift=50); m=pos[:,1]>=0
        ax.plot(pos[m,0]/1e3,pos[m,1]/1e3,color=CD,lw=1.5,ls="--",alpha=.6,label="Decoy" if i==0 else None)
    ax.set_xlabel("Horizontal (km)"); ax.set_ylabel("Altitude (km)")
    ax.set_title("(g) Simulated Trajectories"); ax.legend(loc="upper right"); ax.set_ylim(bottom=-.2)

    ax=fig.add_subplot(gs[3,1])
    np.random.seed(123)
    mb=add_noise(gen_bal(128,.05,v0=300,ang=55))
    mc=add_noise(gen_clutter(128,.05))
    md=add_noise(gen_decoy(128,.05,lift=50))
    sb=srpf_score(mb,.05); sc=srpf_score(mc,.05); sd=srpf_score(md,.05)
    iv_b=innov_2d(mb,.05); iv_c=innov_2d(mc,.05); iv_d=innov_2d(md,.05)
    t=np.arange(128)*.05
    ax.plot(t,np.linalg.norm(iv_b,axis=1),color=CB,lw=1.2,label=f"Bal (σ={sb:.3f})")
    ax.plot(t,np.linalg.norm(iv_c,axis=1),color=CC,lw=1.2,label=f"Cl (σ={sc:.3f})")
    ax.plot(t,np.linalg.norm(iv_d,axis=1),color=CD,lw=1.2,label=f"Dec (σ={sd:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("$||\\delta_k||$ (m)")
    ax.set_title("(h) Innovation Sequences & SRPF Scores"); ax.legend()

    fig.suptitle("SRPF Monte Carlo Validation — X. J. Régent, 2026",fontsize=14,fontweight="bold",y=.97)
    plt.savefig(path,dpi=150,bbox_inches="tight"); print(f"\n  Figure → {path}"); plt.close()


if __name__=="__main__":
    np.random.seed(2026); t0=time.time()
    print("\n"+"="*60+"\n  SRPF Monte Carlo v3\n"+"="*60+"\n")
    e1=exp1(500,128,.05); e2=exp2(2000,128,.05)
    e3=exp3(1500,128,.05,.3); e4=exp4(500,500,.1,100)
    plot_all(e1,e2,e3,e4,"/home/claude/srpf_monte_carlo.png")
    el=time.time()-t0; sl=e1["slopes"]; b=e3["best"]
    print(f"\n{'='*65}\n  SUMMARY\n{'='*65}")
    print(f"  PSD slopes:         {sl[0]:.2f} / {sl[1]:.2f} / {sl[2]:.2f}")
    print(f"  AUC SRPF/Energy:    {e2['auc']:.3f} / {e2['auc_e']:.3f}")
    print(f"  Gain G(τ*):         ×{b['G']:.2f} (P_D={b['g']*100:.0f}%, filt={b['filt']*100:.0f}%)")
    print(f"  Saturation missed:  {np.mean(e4['ms']):.1f} vs {np.mean(e4['mn']):.1f}")
    print(f"  Runtime:            {el:.0f}s\n{'='*65}")
