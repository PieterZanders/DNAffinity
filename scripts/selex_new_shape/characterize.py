import numpy as np, pandas as pd, itertools
from scipy.stats import pearsonr, spearmanr
R='data/'
rc=lambda s:s.translate(str.maketrans('ACGT','TGCA'))[::-1]
def load(p):
    d={}
    for l in open(p):
        t=l.split()
        if 'SHIFT' in l and not t[0].isalpha() or t[0].startswith(('SEQ','SHIFT')): continue
        d[t[0]]=np.array(t[1:],float)
    return pd.DataFrame(d).T
out=[]
for k,a,f in [(4,'selex/tetramer/avg_tetramer.dat','selex/tetramer/fce_tetramer.dat'),(6,'selex/hexamer/avg_hexamer.dat','selex/hexamer/fce_hexamer.dat')]:
    A=load(R+a);F=load(R+f)
    out.append(f'k={k}: avg {A.shape}, fce {F.shape}, NaN avg {A.isna().sum().sum()} fce {F.isna().sum().sum()}, all kmers {len(A)==4**k}')
    M=F.values.reshape(-1,6,6); out.append(f'  fce symmetric max|M-MT|={np.abs(M-M.transpose(0,2,1)).max():.2e}; min eig {np.linalg.eigvalsh((M+M.transpose(0,2,1))/2).min():.3g}')
    # RC relation: shift,tilt flip sign
    s=np.array([-1,1,1,-1,1,1]); idx=[rc(x) for x in A.index]
    out.append(f'  avg RC: corr per col (x vs sign*rc(x)) '+str(np.round([pearsonr(A.values[:,j],s[j]*A.loc[idx].values[:,j])[0] for j in range(6)],3))+f'; max abs diff {np.abs(A.values-s*A.loc[idx].values).max():.3g}')
    out.append('  avg col mean/sd: '+str(A.describe().loc[['mean','std']].round(3).to_dict()))
    if k==4:
        O=load(R+'input/avg_tetramer.dat').loc[A.index]; OF=load(R+'input/fce_tetramer.dat').loc[A.index]
        out.append('  vs data/input avg corr per col: '+str(np.round([pearsonr(A.values[:,j],O.values[:,j])[0] for j in range(6)],3)))
        out.append(f'  vs data/input fce: corr diag '+str(np.round([pearsonr(F.values[:,j],OF.values[:,j])[0] for j in range(0,36,7)],3)))
        A4=A
    else:
        c=A4.loc[[x[1:5] for x in A.index]].values
        out.append('  hexamer avg vs central tetramer avg corr: '+str(np.round([pearsonr(A.values[:,j],c[:,j])[0] for j in range(6)],3)))
d=pd.read_csv(R+'selex/Gata4/4.txt',sep='\t')
out.append(f'4.txt: {len(d)} rows of {4**10} 10-mers ({len(d)/4**10:.1%}); count==1 {np.mean(d.ObservedCount==1):.1%}; >=10 {np.sum(d.ObservedCount>=10)}')
r=d.Affinity/(d.ObservedCount/d.ExpectedCount)**0.25; out.append(f'  Aff/(obs/exp)^.25: mean {r.mean():.4f} sd {r.std():.2e}')
out.append(f'  SE*sqrt(n)/Aff: median {(d.SE*np.sqrt(d.ObservedCount)/d.Affinity).median():.3f}')
v=dict(zip(d.Kmer,d.Affinity)); n=dict(zip(d.Kmer,d.ObservedCount))
for th in [1,10,30]:
    pr=[(np.log(v[x]),np.log(v[rc(x)])) for x in v if x<rc(x) and rc(x) in v and n[x]>=th and n[rc(x)]>=th]
    if len(pr)>5: p=np.array(pr); out.append(f'  fwd vs RC log-aff r (both counts>={th}, n={len(p)}): {pearsonr(p[:,0],p[:,1])[0]:.3f}')
t=pd.read_csv(R+'selex/Gata4/SELEX_test.txt',sep='\t'); out.append(f'SELEX_test: {len(t)} rows, len {t.ID_REF.str.len().unique()}, VALUE range {t.VALUE.min():.3g}-{t.VALUE.max():.3g}, WEIGHT uniq {t.WEIGHT.unique()[:5]}')
inn=t.ID_REF.isin(v.keys()); out.append(f'  in 4.txt: {inn.mean():.1%}; VALUE==4.txt Aff for overlap: {np.allclose(t.VALUE[inn],[v[x] for x in t.ID_REF[inn]])}')
inrc=t.ID_REF.map(rc).isin(v.keys()); out.append(f'  RC in 4.txt: {inrc.mean():.1%}; counts of test kmers: '+str(pd.Series([n.get(x,0) for x in t.ID_REF]).describe().round(1).to_dict()))
print('\n'.join(out)); open('outputs/selex_new_shape/characterization.txt','w').write('\n'.join(out))
